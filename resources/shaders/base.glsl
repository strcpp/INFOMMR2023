#version 330

#if defined VERTEX_SHADER

in vec3  in_position;
in vec3  in_normal;

flat out vec3 normal;
flat out vec3 fragPos;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

void main() {

    normal = mat3(transpose(inverse(model))) * normalize(in_normal);
    fragPos = vec3(model * vec4(in_position, 1)); 

    gl_Position = projection * view * model * vec4(in_position, 1);
}

#elif defined FRAGMENT_SHADER

out vec4 f_color;

struct Light {
    vec3 position;
    vec3 Ia; 
    vec3 Id; 
    vec3 Is;
};

flat in vec3 normal;
flat in vec3 fragPos;

uniform sampler2D Texture;
uniform Light light;
uniform vec3 camPos;
// uniform bool useTexture;

// simple phong model
vec3 calculateLighting() {
    vec3 Normal  = normalize(normal);
    vec3 ambient = light.Ia;
    
    vec3 dir = normalize(light.position - fragPos);
    vec3 diffuse = light.Id * max(0, dot(dir, normal));
    
    vec3 viewDir = normalize(camPos - fragPos);
    vec3 reflectDir = reflect(-dir, Normal);
    float spec = pow(max(0, dot(viewDir, reflectDir)), 32);
    vec3 specular = light.Is * spec;
    
    // Attenuation
    float constant = 1.0;
    float linear = 0.09;
    float quadratic = 0.032;
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (constant + linear * distance + quadratic * (distance * distance));
    
    return (ambient + diffuse + specular) * attenuation;
}

void main() {
    float gamma = 2.2;

    vec3 color = vec3(1.0, 0.0, 0.0);    
    // if(useTexture) {
    //     color = texture(Texture, tex_coords).rgb; // Note: `tex_coords` is commented out in your original shader
    // }

    color = pow(color, vec3(gamma));
    color = color * calculateLighting();
    color = pow(color, 1 / vec3(gamma));

    f_color = vec4(color, 1.0);
}

#endif