#version 330 core
out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D bumpMap;
uniform samplerCube cubeMap;
uniform vec4 deepColor;
uniform vec4 shallowColor;
uniform float fresnelPower;
uniform float hdrMultiplier;

void main()
{
    vec3 normal = texture(bumpMap, TexCoord).rgb;
    vec3 viewDir = normalize(camera_position - vec3(gl_FragCoord));
    float fresnel = pow(max(dot(viewDir, normal), 0.0), fresnelPower);
    vec4 color = mix(shallowColor, deepColor, fresnel);
    FragColor = color * hdrMultiplier;
}
