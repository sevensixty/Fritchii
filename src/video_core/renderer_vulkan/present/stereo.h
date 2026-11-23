// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>

#include "common/math_util.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

/// Builds a side-by-side stereo image from one colour image and the frame's depth buffer.
/// The output holds the left eye in its left half and the right eye in its right half.
class StereoPass final {
public:
    explicit StereoPass(const Device& device, MemoryAllocator& allocator, size_t image_count,
                        VkExtent2D extent);
    ~StereoPass();

    [[nodiscard]] VkExtent2D Extent() const noexcept {
        return m_extent;
    }

    /// depth_view may be null, which gives both eyes the same image.
    void Draw(const Device& device, Scheduler& scheduler, size_t image_index,
              VkImage* inout_image, VkImageView* inout_image_view, VkImageView depth_view,
              const std::array<f32, 4>& depth_region, const Common::Rectangle<f32>& crop);

private:
    struct Image {
        vk::DescriptorSets descriptor_sets{};
        vk::Framebuffer framebuffer{};
        vk::Image image{};
        vk::ImageView image_view{};
    };

    void UpdateDescriptorSets(const Device& device, VkImageView color_view,
                              VkImageView depth_view, size_t image_index);
    void ClearImages(Scheduler& scheduler);

    std::vector<Image> m_images{};
    const VkExtent2D m_extent;
    const u32 m_image_count;
    vk::ShaderModule m_vertex_shader{};
    vk::ShaderModule m_fragment_shader{};
    vk::DescriptorPool m_descriptor_pool{};
    vk::DescriptorSetLayout m_descriptor_set_layout{};
    vk::PipelineLayout m_pipeline_layout{};
    vk::Pipeline m_pipeline{};
    vk::RenderPass m_renderpass{};
    vk::Sampler m_color_sampler{};
    vk::Sampler m_depth_sampler{};
    bool m_images_ready{};
};

} // namespace Vulkan
