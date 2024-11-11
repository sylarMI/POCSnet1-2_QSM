matrix_size = [64 64 64]; %%160 160 160

voxel_size = [0.8594 0.8594 1];
N = matrix_size;

[ky,kx,kz] = meshgrid(-N(1)/2:N(1)/2-1, -N(2)/2:N(2)/2-1, -N(3)/2:N(3)/2-1);

kx = (kx / max(abs(kx(:)))) / voxel_size(1);
ky = (ky / max(abs(ky(:)))) / voxel_size(2);
kz = (kz / max(abs(kz(:)))) / voxel_size(3);


k2 = kx.^2 + ky.^2 + kz.^2;
D =  1/3 - kz.^2 ./ (k2+eps) ;
kernel = fftshift(D);

for i=1:1000
    b = -.5+1*rand(1,1);
    b1 = -.5+1*rand(1,1);
    std = 0.2; %0.02;
    
    E = [2 0 matrix_size(1) matrix_size(2) matrix_size(3) 0 0 0 0 0 0;
        randi([1, 2], 1) normrnd(0,std,1) -1+2*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -0.9+1.8*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -0.8+1.6*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -0.7+1.4*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -0.6+1.2*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -0.5+1*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -1+2*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.9+1.8*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.8+1.6*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.7+1.4*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.6+1.2*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.5+1*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.5+1*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.5+1*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) b b b -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);              % simulate a cube
        randi([1, 2], 1) normrnd(0,std,1) -.5+1*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.5+1*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.5+1*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -1+2*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -0.9+1.8*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -0.8+1.6*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -0.7+1.4*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -0.6+1.2*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -0.5+1*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -1+2*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.9+1.8*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.8+1.6*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.7+1.4*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.6+1.2*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.5+1*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.5+1*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.5+1*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) b1 b1 b1 -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);              % simulate a cube
        randi([1, 2], 1) normrnd(0,std,1) -.5+1*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.5+1*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.5+1*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);
        randi([1, 2], 1) normrnd(0,std,1) -.5+1*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3);   
        randi([1, 2], 1) normrnd(0,std,1) -.5+1*rand(1,3) -0.9+1.8*rand(1,3) -pi+2*pi*rand(1,3)];
 
    ph = phantom3d_shapes(E,matrix_size);

    save_nii(make_nii(ph,voxel_size),['/Data/Simu_POCSnet2/qsm/qsm' num2str(i) '.nii.gz']);
    
    phs_tissue = ifftn(kernel.*fftn(ph));
    noise_std = max(phs_tissue(:))/100;
    phs_tissue = phs_tissue + normrnd(0,noise_std,N(1),N(2),N(3));
    save_nii(make_nii(phs_tissue,voxel_size),['/Data/Simu_POCSnet2/phs/phs' num2str(i) '.nii.gz']);

end