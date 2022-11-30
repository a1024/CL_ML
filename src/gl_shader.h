#version 460 core
layout(local_size_x=8, local_size_y=4, local_size_z=1) in;
layout(r32f, binding=0) uniform image2D screen;
void main()
{
	//https://www.youtube.com/watch?v=nF4X9BIUzx0
	ivec2 pos=ivec2(gl_GlobalInvocationID.xy);
	//float pixel=0.075;
	vec4 pixel=vec4(0.075, 0.133, 0.173, 1.);

	ivec2 dims=imageSize(screen);
	float
		x=float(dims.x-pos.x*2)/dims.x,
		y=float(dims.y-pos.y*2)/dims.y;
	float fov=90.;
	vec3 cam_o=vec3(0., 0., -tan(fov/2.)), ray_o=vec3(x, y, 0.), ray_d=normalize(ray_o-cam_o);
	vec3 sphere_c=vec3(0., 0., -5.);
	float sphere_r=1.;
	vec3 o_c=ray_o-sphere_c;
	float b=dot(ray_d, o_c), c=dot(o_c, o_c)-sphere_r*sphere_r, state=b*b-c;
	if(state>=0.)
		pixel=vec4((normalize(ray_o+ray_d*(-b+sqrt(state))-sphere_c)+1.)/2., 1.);
	imageStore(screen, pos, pixel);
}