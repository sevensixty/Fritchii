// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Side-by-side stereo from one colour image and its depth buffer (depth-image-based rendering).
// Each eye is made by a backward warp: for an output pixel, search for the nearest source pixel
// whose depth-driven shift lands on it.
// The disparity follows the clip-space form: shift = dir * separation * (conv / z - 1) / 2 eye
// widths, with conv / z written as w / w_conv where w ~ near / z is read from device depth.

#version 460

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 color;

layout(binding = 0) uniform sampler2D color_texture;
layout(binding = 1) uniform sampler2D depth_texture;

layout(push_constant) uniform PushConstants {
    vec4 crop;          // left, top, right, bottom of the source, normalized
    float separation;   // total background disparity, fraction of eye width
    float inv_conv;     // near / convergence distance: w of the screen plane
    int depth_mode;     // 0 auto, 1 standard (near = 0), 2 reversed (near = 1), 3 show depth
    int flags;          // bit 0: depth present, bit 1: swap eyes, bit 2: flip depth vertically
};

// Candidate disparities searched per pixel, from nearest to farthest.
const int STEPS = 48;
// Pop-out is not divergence: allow a little more crossed disparity than uncrossed, but keep it
// small, because the gap a near object leaves behind is filled by stretched background.
const float LIMIT_NEAR = 0.04;

bool reversed;

float InverseDepth(vec2 uv) {
    if ((flags & 4) != 0) {
        uv.y = 1.0 - uv.y;
    }
    const float d = textureLod(depth_texture, uv, 0.0).r;
    return reversed ? d : 1.0 - d;
}

// Disparity of the source pixel at uv, toward this eye's near direction, in eye widths.
float Parallax(vec2 uv) {
    const float ratio = InverseDepth(uv) / inv_conv;
    return clamp(separation * (ratio - 1.0) * 0.5, -separation * 0.5, LIMIT_NEAR);
}

bool DetectReversed() {
    if (depth_mode == 1 || depth_mode == 2) {
        return depth_mode == 2;
    }
    // Most of a frame is far away: far depth reads near 1 in a standard buffer, near 0 reversed.
    float sum = 0.0;
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            sum += textureLod(depth_texture, vec2(0.25 + 0.25 * x, 0.25 + 0.25 * y), 0.0).r;
        }
    }
    return sum / 9.0 < 0.5;
}

void main() {
    const bool right_half = frag_uv.x >= 0.5;
    const float eye_u = right_half ? (frag_uv.x - 0.5) * 2.0 : frag_uv.x * 2.0;
    int eye = right_half ? 1 : 0;
    if ((flags & 2) != 0) {
        eye = 1 - eye;
    }
    // Left eye (+1): near content moves right, far content moves left.
    const float dir = eye == 0 ? 1.0 : -1.0;

    const vec2 crop_size = crop.zw - crop.xy;
    const vec2 target = crop.xy + vec2(eye_u, frag_uv.y) * crop_size;
    vec2 source = target;
    if ((flags & 1) != 0 && depth_mode == 3) {
        reversed = DetectReversed();
        color = vec4(vec3(clamp(InverseDepth(target) / inv_conv, 0.0, 1.0)), 1.0);
        return;
    }
    if ((flags & 1) != 0 && separation > 0.0) {
        reversed = DetectReversed();
        // Find the nearest source pixel that lands on this target: march the candidate
        // disparity p from the near limit to the far limit and stop at the first surface whose
        // own disparity reaches p. Searching near first keeps foreground objects solid instead
        // of letting background show through them.
        const float p_near = LIMIT_NEAR;
        const float p_far = -separation * 0.5;
        const float step = (p_near - p_far) / float(STEPS);
        float prev_p = p_near;
        float prev_f = Parallax(vec2(target.x - dir * prev_p * crop_size.x, target.y)) - prev_p;
        float found = p_far;
        for (int i = 1; i <= STEPS; ++i) {
            const float p = p_near - step * float(i);
            const float f = Parallax(vec2(target.x - dir * p * crop_size.x, target.y)) - p;
            if (f >= 0.0) {
                // Interpolate between the last miss and this hit.
                const float t = prev_f < 0.0 ? prev_f / (prev_f - f) : 0.0;
                found = mix(prev_p, p, t);
                break;
            }
            prev_p = p;
            prev_f = f;
        }
        source.x = clamp(target.x - dir * found * crop_size.x, crop.x, crop.z);
    }
    color = vec4(textureLod(color_texture, source, 0.0).rgb, 1.0);
}
