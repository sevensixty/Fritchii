// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Side-by-side stereo from one colour image and its depth buffer (depth-image-based rendering).
// Each eye is made by a backward warp: for an output pixel, find the source pixel whose
// depth-driven shift lands on it, by fixed-point iteration on the depth at the current guess.
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
    int depth_mode;     // 0 auto, 1 standard (near = 0), 2 reversed (near = 1)
    int flags;          // bit 0: depth present, bit 1: swap eyes
};

const int ITERATIONS = 4;
// Pop-out is not divergence: allow more crossed disparity than uncrossed (fractions of eye width).
const float LIMIT_NEAR = 0.15;

bool reversed;

float InverseDepth(vec2 uv) {
    const float d = textureLod(depth_texture, uv, 0.0).r;
    return reversed ? d : 1.0 - d;
}

float Shift(vec2 uv, float dir) {
    const float ratio = InverseDepth(uv) / inv_conv;
    const float s = dir * separation * (ratio - 1.0) * 0.5;
    return clamp(s, -LIMIT_NEAR, LIMIT_NEAR);
}

bool DetectReversed() {
    if (depth_mode != 0) {
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
    if ((flags & 1) != 0 && separation > 0.0) {
        reversed = DetectReversed();
        for (int i = 0; i < ITERATIONS; ++i) {
            source.x = target.x - Shift(source, dir) * crop_size.x;
        }
    }
    color = vec4(textureLod(color_texture, source, 0.0).rgb, 1.0);
}
