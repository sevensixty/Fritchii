// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>

#include "common/settings.h"
#include "video_core/host_shaders/vulkan_stereo_sbs_frag_spv.h"
#include "video_core/host_shaders/vulkan_stereo_sbs_vert_spv.h"
#include "video_core/renderer_vulkan/present/stereo.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

namespace {

constexpr VkFormat OutputFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

// Matches the push constant block in vulkan_stereo_sbs.frag.
struct StereoPushConstants {
    std::array<f32, 4> crop;
    f32 separation;
    f32 inv_conv;
    s32 depth_mode;
    s32 flags;
};
static_assert(sizeof(StereoPushConstants) == 32);

} // Anonymous namespace

StereoPass::StereoPass(const Device& device, MemoryAllocator& allocator, size_t image_count,
                       VkExtent2D extent)
    : m_extent(extent), m_image_count(u32(image_count)) {
    for (u32 i = 0; i < m_image_count; i++) {
        Image& image = m_images.emplace_back();
        image.image = CreateWrappedImage(allocator, m_extent, OutputFormat);
        image.image_view = CreateWrappedImageView(device, image.image, OutputFormat);
    }
    m_renderpass = CreateWrappedRenderPass(device, OutputFormat);
    for (auto& image : m_images) {
        image.framebuffer =
            CreateWrappedFramebuffer(device, m_renderpass, image.image_view, m_extent);
    }
    m_color_sampler = CreateWrappedSampler(device, VK_FILTER_LINEAR);
    m_depth_sampler = CreateWrappedSampler(device, VK_FILTER_NEAREST);
    m_vertex_shader = CreateWrappedShaderModule(device, VULKAN_STEREO_SBS_VERT_SPV);
    m_fragment_shader = CreateWrappedShaderModule(device, VULKAN_STEREO_SBS_FRAG_SPV);

    // 2 descriptors, 1 descriptor set per image
    m_descriptor_pool = CreateWrappedDescriptorPool(device, 2 * m_image_count, m_image_count);
    m_descriptor_set_layout = CreateWrappedDescriptorSetLayout(
        device, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER});
    VkDescriptorSetLayout layout = *m_descriptor_set_layout;
    for (auto& image : m_images) {
        image.descriptor_sets = CreateWrappedDescriptorSets(m_descriptor_pool, {layout});
    }

    const VkPushConstantRange push_range{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = sizeof(StereoPushConstants),
    };
    m_pipeline_layout = device.GetLogical().CreatePipelineLayout(VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = m_descriptor_set_layout.address(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    });
    m_pipeline = CreateWrappedPipeline(device, m_renderpass, m_pipeline_layout,
                                       std::tie(m_vertex_shader, m_fragment_shader));
}

StereoPass::~StereoPass() = default;

void StereoPass::UpdateDescriptorSets(const Device& device, VkImageView color_view,
                                      VkImageView depth_view, size_t image_index) {
    Image& image = m_images[image_index];
    std::vector<VkDescriptorImageInfo> image_infos;
    std::vector<VkWriteDescriptorSet> updates;
    image_infos.reserve(2);
    updates.push_back(CreateWriteDescriptorSet(image_infos, *m_color_sampler, color_view,
                                               image.descriptor_sets[0], 0));
    // Without depth, the colour image stands in so the binding stays valid; the shader skips it.
    updates.push_back(CreateWriteDescriptorSet(image_infos, *m_depth_sampler,
                                               depth_view ? depth_view : color_view,
                                               image.descriptor_sets[0], 1));
    device.GetLogical().UpdateDescriptorSets(updates, {});
}

void StereoPass::ClearImages(Scheduler& scheduler) {
    if (m_images_ready) {
        return;
    }
    scheduler.Record([&](vk::CommandBuffer cmdbuf) {
        for (auto& image : m_images) {
            ClearColorImage(cmdbuf, *image.image);
        }
    });
    scheduler.Finish();
    m_images_ready = true;
}

void StereoPass::Draw(const Device& device, Scheduler& scheduler, size_t image_index,
                      VkImage* inout_image, VkImageView* inout_image_view,
                      VkImageView depth_view, const Common::Rectangle<f32>& crop) {
    const Image& image{m_images[image_index]};
    const VkImage input_image{*inout_image};
    const VkImage output_image{*image.image};
    const VkDescriptorSet descriptor_set{image.descriptor_sets[0]};
    const VkFramebuffer framebuffer{*image.framebuffer};
    const VkRenderPass renderpass{*m_renderpass};
    const VkPipeline pipeline{*m_pipeline};
    const VkPipelineLayout layout{*m_pipeline_layout};
    const VkExtent2D extent{m_extent};

    const auto& values = Settings::values;
    const StereoPushConstants push{
        .crop = {crop.left, crop.top, crop.right, crop.bottom},
        // The setting is total disparity as a percentage of the screen; the shader works in
        // eye widths, which for side-by-side is the same fraction.
        .separation = f32(values.stereo_separation.GetValue()) / 100.0f,
        .inv_conv = 1.0f / f32((std::max)(values.stereo_convergence.GetValue(), 1)),
        .depth_mode = s32(values.stereo_depth.GetValue()),
        .flags = (depth_view ? 1 : 0) | (values.stereo_swap_eyes.GetValue() ? 2 : 0) |
                 (values.stereo_flip_depth.GetValue() ? 4 : 0),
    };

    ClearImages(scheduler);
    UpdateDescriptorSets(device, *inout_image_view, depth_view, image_index);

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([=](vk::CommandBuffer cmdbuf) {
        // The depth buffer was last written as an attachment by the game's draws.
        const VkMemoryBarrier depth_barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, depth_barrier);
        TransitionImageLayout(cmdbuf, input_image, VK_IMAGE_LAYOUT_GENERAL);
        TransitionImageLayout(cmdbuf, output_image, VK_IMAGE_LAYOUT_GENERAL);
        BeginRenderPass(cmdbuf, renderpass, framebuffer, extent);
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, descriptor_set, {});
        cmdbuf.PushConstants(layout, VK_SHADER_STAGE_FRAGMENT_BIT, push);
        cmdbuf.Draw(3, 1, 0, 0);
        cmdbuf.EndRenderPass();
        TransitionImageLayout(cmdbuf, output_image, VK_IMAGE_LAYOUT_GENERAL);
    });

    *inout_image = *image.image;
    *inout_image_view = *image.image_view;
}

} // namespace Vulkan
