#ifdef GL_ES
precision mediump float;
#endif

uniform sampler2D u_tex0;
uniform vec2 u_resolution;
uniform vec2 u_tex0Resolution;
uniform vec2 u_mouse;

uniform float scale;

void main(){
    float s = 1.0 + scale;
    vec2 tex_resolution = u_tex0Resolution.xy / s;
    vec2 st = gl_FragCoord.xy / tex_resolution;
    vec2 offset = clamp(-0.05 + 1.1 * u_mouse / u_resolution, 0.0, 1.0);
    st += offset * (tex_resolution - u_resolution) / (tex_resolution);
    gl_FragColor = texture2D(u_tex0, vec2(st.x, st.y));
}
