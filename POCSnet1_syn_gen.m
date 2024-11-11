
ori_size = [256 256 98];
matrix_size =  [256 256 98];

voxel_size = [0.9375 0.9375 1.5];


N = matrix_size;
msk_inv = zeros(N);


[ky,kx,kz] = meshgrid(-N(1)/2:N(1)/2-1, -N(2)/2:N(2)/2-1, -N(3)/2:N(3)/2-1);

kx = (kx / max(abs(kx(:)))) / voxel_size(1);
ky = (ky / max(abs(ky(:)))) / voxel_size(2);
kz = (kz / max(abs(kz(:)))) / voxel_size(3);

k2 = kx.^2 + ky.^2 + kz.^2;
kernel = fftshift( 1/3 - kz.^2 ./ (k2 + eps) );

qsm = niftiread('/data/cheat.nii');
ph_msk = niftiread('/data/msk.nii');
qsm = imresize3(qsm,matrix_size);
qsm(isnan(qsm))=0;
ph_msk = imbinarize(imresize3(ph_msk,matrix_size));

[xx,yy,zz] = ndgrid(-16:16);
nhood = sqrt(xx.^2+yy.^2+zz.^2)<= 15;
nhood1 = sqrt(xx.^2+yy.^2+zz.^2)<= 15;
erode_msk = imerode(ph_msk,nhood);
dilate_msk = imdilate(ph_msk,nhood1);

msk_inv(dilate_msk==0) = 1;


for i=1:300

    % hemorrhage simulation
    E = [randi([1, 2], 1) normrnd(0.30,0.05,1) -0.1+0.05*rand(1,3) -0.4+.8*rand(1,2) -0.6+ 1.2*rand(1,1) -180+360*rand(1,3);
        randi([1, 2], 1) normrnd(0.25,0.05,1) -0.1+0.05*rand(1,3) -0.4+.8*rand(1,2) -0.6+ 1.2*rand(1,1) -180+360*rand(1,3)]; % with eroded mask and greater values
        
    hm = phantom3d_shapes(E,matrix_size);

    qsm_com = qsm+hm.*erode_msk;
    
    phs_tissue = real(ifftn(kernel.*fftn(qsm_com)));
    noise_std = max(phs_tissue(:))/50;
    phs_tissue = (phs_tissue + normrnd(0,noise_std,N(1),N(2),N(3))).*ph_msk;

   % background resource simulation
   r = randi([1,2],1);
   pn = randi([0,1],1);
   if r==1
        E_bg = [randi([1, 2], 1) normrnd(6.28,0.5,1) 0.08+0.03*rand(1,3) -0.6-0.1*rand(1,3) -180+360*rand(1,3);
            randi([1, 2], 1) normrnd(6.28,0.5,1) 0.08+0.03*rand(1,3) 0.6+0.1*rand(1,3) -180+360*rand(1,3);
            randi([1, 2], 1) normrnd(6.28,0.5,1) 0.08+0.03*rand(1,3) -0.6-0.1*rand(1,2) 0.6+0.1*rand(1,1) -180+360*rand(1,3);
            randi([1, 2], 1) normrnd(-6.28,0.5,1) 0.08+0.03*rand(1,3) 0.6+0.1*rand(1,1) -0.6-0.1*rand(1,1) 0.6+0.1*rand(1,1) -180+360*rand(1,3)];
   else
       E_bg = [randi([1, 2], 1) normrnd(6.28,0.5,1) 0.08+0.03*rand(1,3) -0.6-0.1*rand(1,3) -180+360*rand(1,3);
            randi([1, 2], 1) normrnd(6.28,0.5,1) 0.08+0.03*rand(1,3) -0.6-0.1*rand(1,2) 0.6+0.1*rand(1,1) -180+360*rand(1,3);
            randi([1, 2], 1) normrnd(-6.28,0.5,1) 0.08+0.03*rand(1,3) 0.6+0.1*rand(1,1) -0.6-0.1*rand(1,1) 0.6+0.1*rand(1,1) -180+360*rand(1,3)];
   end

    bg_qsm = phantom3d_shapes(E_bg,matrix_size);
    
    bg_phs = real(ifftn(kernel.*fftn(bg_qsm.*msk_inv)));
    total_phs = bg_phs+phs_tissue;
    total_phs_c = total_phs.*ph_msk;

    save_nii(make_nii(total_phs,voxel_size),['/data/simu_POCSnet1/phs_total/phs' num2str(i) '.nii.gz']);
    save_nii(make_nii(qsm_com,voxel_size),['/data/simu_POCSnet1/qsm/qsm' num2str(i) '.nii.gz']);
    save_nii(make_nii(hm,voxel_size),['/data/simu_POCSnet1/hemr/qsm' num2str(i) '.nii.gz']);
    save_nii(make_nii(phs_tissue,voxel_size),['/data/simu_POCSnet1/phs_tissue/phs' num2str(i) '.nii.gz']);
end
