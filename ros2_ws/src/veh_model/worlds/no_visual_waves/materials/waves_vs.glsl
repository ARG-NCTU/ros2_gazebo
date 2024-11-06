#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;

uniform mat4 world_matrix;
uniform mat4 worldviewproj_matrix;
uniform vec3 camera_position;

out vec2 TexCoord;

void main()
{
    gl_Position = worldviewproj_matrix * world_matrix * vec4(position, 1.0);
    TexCoord = texCoord;
}
