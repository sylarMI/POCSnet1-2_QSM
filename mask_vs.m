
voxel_size = [0.9375 0.9375 1.5];
N= [64 64 64];


for i = 1:10
    msk = load_nii(['/data/simu_POCSnt1/patch48/mask/msk' num2str(i) '.nii.gz']);
    msk = double(msk.img);
    msk = padarray(msk,[8,8,8],0,'both');
    
    msk_arr = zeros([N,12]);
    for ker_rad = 1:12
        % make spherical/ellipsoidal convolution kernel (ker)
        rx = round(ker_rad/vox(1));
        ry = round(ker_rad/vox(2));
        rz = round(ker_rad/vox(3));
        rx = max(rx,2);
        ry = max(ry,2);
        rz = max(rz,2);
        % rz = ceil(ker_rad/vox(3));
        [X,Y,Z] = ndgrid(-rx:rx,-ry:ry,-rz:rz);
        h = (X.^2/rx^2 + Y.^2/ry^2 + Z.^2/rz^2 <= 1);
        ker = h/sum(h(:));
        
        mask_ero = zeros(N);
        mask_tmp = convn(msk,ker,'same');
        mask_ero(mask_tmp > 0.999999) = 1;
        msk_arr(:,:,:,13-ker_rad) = mask_ero;
    end
    msk_c = zeros([N,12]);
    for k = 1:12
        if k == 1
            msk_c(:,:,:,k) = msk_arr(:,:,:,k);
        else
            msk_c(:,:,:,k) = msk_arr(:,:,:,k)-msk_arr(:,:,:,k-1);
        end
    end
    
    save_nii(make_nii(msk_c, voxel_size), ['/data/simu_POCSnt1/patch48/mask_arr/msk_c' num2str(i) '.nii.gz']);
    save_nii(make_nii(msk_arr, voxel_size), ['/data/simu_POCSnt1/patch48/mask_arr/msk' num2str(i) '.nii.gz']);
       
end