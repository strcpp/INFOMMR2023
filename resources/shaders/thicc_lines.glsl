#version 330

// reference: https://www.gamedev.net/forums/topic/713244-opengl-4-thick-lines/

#if defined VERTEX_SHADER

in vec3 position;
uniform mat4 model;
uniform mat4 projection;

uniform mat4 view;


void main()
{
    gl_Position = projection*view*model*vec4(position, 1.0);
}

#elif defined FRAGMENT_SHADER
out vec4 FragColor;
uniform vec4 color;


void main()
{
    FragColor = vec4(color);
}

#elif defined GEOMETRY_SHADER

layout (lines) in;                              // now we can access 2 vertices
layout (triangle_strip, max_vertices = 4) out;  // always (for now) producing 2 triangles (so 4 vertices)


uniform int img_width;
uniform int img_height;
uniform float line_thickness;

vec2  u_viewportSize = vec2(img_width, img_height);
 float u_thickness =  line_thickness;



void main()
{
    vec4 p1 = gl_in[0].gl_Position;
    vec4 p2 = gl_in[1].gl_Position;
    vec2 dir    = normalize((p2.xy/p2.w - p1.xy/p1.w) * u_viewportSize);
    vec2 offset = vec2(-dir.y, dir.x) * u_thickness / u_viewportSize;
    gl_Position = p1 + vec4(offset.xy * p1.w, 0.0, 0.0);
    EmitVertex();
    gl_Position = p1 - vec4(offset.xy * p1.w, 0.0, 0.0);
    EmitVertex();
    gl_Position = p2 + vec4(offset.xy * p2.w, 0.0, 0.0);
    EmitVertex();
    gl_Position = p2 - vec4(offset.xy * p2.w, 0.0, 0.0);
    EmitVertex();
    EndPrimitive();
}

#endif