#version 330

#if defined VERTEX_SHADER

in vec3 position;
out vec4 clipCoords;

void main()
{
    gl_Position = vec4(position, 1.0);
    clipCoords = gl_Position;
}

#elif defined FRAGMENT_SHADER

out vec4 FragColor;
in vec4 clipCoords;

uniform samplerCube u_texture_skybox;
uniform mat4 m_invProjView;

void main()
{
    vec4 worldCoords = m_invProjView * clipCoords;
    vec3 texCubeCoord = normalize(worldCoords.xyz / worldCoords.w);
    FragColor = texture(u_texture_skybox, texCubeCoord);
}

#endif