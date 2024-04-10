function [sphere_points] = makesphere(center,radius,density)
%This function will generate points for a filled spheres given radius and center
radius_plot = linspace(0.1,radius, density);
theta = linspace(0,2 .*pi, density); 
phi = linspace(0,pi,density); 
sphere_points = [];
for i=1:density
phi_rand = phi(randperm(length(phi))); 
theta_rand = theta(randperm(length(theta)));
x = radius_plot(i).*sin(phi_rand)' .* cos(theta_rand)';
y = radius_plot(i).*sin(phi_rand)' .* sin(theta_rand)';
z = radius_plot(i).*cos(phi_rand)';
points_iter = horzcat(x,y,z); 
sphere_points = vertcat(sphere_points,points_iter); 
end

n = size(sphere_points); 

for i=1:n(1)
    sphere_points(i,:) = sphere_points(i,:) + center;
end


scatter3(sphere_points(:,1),sphere_points(:,2), sphere_points(:,3))
