#version 330

#if defined VERTEX_SHADER

in vec3 position;
in vec2  in_texcoord_0;

out vec3 fragPos;

out vec2 texcoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
    texcoords = in_texcoord_0;
    fragPos = vec3(model * vec4(position, 1.0)); // Convert position to world space
}

#elif defined FRAGMENT_SHADER

in vec3 fragPos;
in vec2 texcoords;
// uniform vec3 color;
uniform vec3 cameraPos;

// uniform float val; 

out vec4 fragColor;

void main() {
    float u = texcoords.x * 20.0;
    float v = texcoords.y * 20.0;

    // Interpolate between blue and white based on the wave value
    vec3 color = vec3(1.0, 1.0, 1.0);;

    // Fog effect
    vec3 fogColor = vec3(0.9, 0.9, 0.9);
    float fogStart = 10.0;
    float fogEnd = 100.0;
    float distance = length(cameraPos - fragPos);
    float fogFactor = clamp((fogEnd - distance) / (fogEnd - fogStart), 0.0, 1.0);

    // Blend the color with the fog color
    vec3 finalColor = mix(fogColor, color, fogFactor);

    // Grid lines
    float gridSpacing = 0.1;
    float gridThickness = 0.005;
    float dotSpacing = 0.02;

    float gridX = float(mod(u, gridSpacing) < gridThickness);
    float gridY = float(mod(v, gridSpacing) < gridThickness);
    float dotPattern = float(mod(u, dotSpacing) + mod(v, dotSpacing) < dotSpacing);

    float lineAlpha = 1.0 - smoothstep(10.0, 50.0, distance); 
    
    if ((gridX > 0.0 || gridY > 0.0) && dotPattern > 0.0) {
        vec3 lineColor = vec3(0.7, 0.7, 0.7);
        finalColor = mix(finalColor, lineColor, lineAlpha);
    }

    fragColor = vec4(finalColor, 1.0);

}

#endif