// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#version 460

out gl_PerVertex {
    vec4 gl_Position;
};

const vec2 vertices[3] = vec2[3](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));

layout(location = 0) out vec2 frag_uv;

void main() {
    const vec2 vertex = vertices[gl_VertexIndex];
    gl_Position = vec4(vertex, 0.0, 1.0);
    frag_uv = (vertex + 1.0) * 0.5;
}
