

patch_size = [48 48 48];
crop_size = [216 216 98];
spatial_res = [0.9375 0.9375 1.5];

z=1;
msk_raw = load('/data/02_Wisnieff_Liu/msk.mat');
msk_raw = double(msk_raw.Mask);
msk_r = msk_raw(21:236,21:236,:,:);
msk = msk(21:236,21:236,:,:);

for i= 1:600

% for simulated data>>>>>>>>>>>>>>
     
    phs_tissue_raw = load_nii(['/data/simu_POCSnet1/phs_tissue/phs' num2str(i) '.nii.gz']);
    phs_total_raw = load_nii(['/data/simu_POCSnet1/phs_total/phs' num2str(i) '.nii.gz']);
    qsm_raw = load_nii(['/data/simu_POCSnet1/qsm/qsm' num2str(i) '.nii.gz']);
    
    phs_total_raw = phs_total_raw.img;
    phs_tissue_raw = phs_tissue_raw.img;
    qsm_raw = qsm_raw.img;

    phs_total = phs_total_raw(21:236,21:236,:);
    phs_tissue = phs_tissue_raw(21:236,21:236,:);
    qsm = qsm_raw(21:236,21:236,:);
    
    for c=1:110 
        rnd1 = randi([1 crop_size(1)-patch_size(1)],1,1);
        rnd2 = randi([1 crop_size(2)-patch_size(2)],1,1);
        rnd3 = randi([1 crop_size(3)-patch_size(3)],1,1);
        msk_c = msk(rnd1:rnd1+patch_size(1)-1,rnd2:rnd2+patch_size(2)-1,rnd3:rnd3+patch_size(3)-1,:);

        phs_total_c = phs_total(rnd1:rnd1+patch_size(1)-1,rnd2:rnd2+patch_size(2)-1,rnd3:rnd3+patch_size(3)-1);
        phs_tissue_c = phs_tissue(rnd1:rnd1+patch_size(1)-1,rnd2:rnd2+patch_size(2)-1,rnd3:rnd3+patch_size(3)-1);
        qsm_c = qsm(rnd1:rnd1+patch_size(1)-1,rnd2:rnd2+patch_size(2)-1,rnd3:rnd3+patch_size(3)-1);
                
        save_nii(make_nii(msk_rc,spatial_res),['/data/simu_POCSnt1/patch48/mask/msk' num2str(z) '.nii.gz']);
        save_nii(make_nii(phs_total_c,spatial_res),['/data/simu_POCSnt1/patch48/phs_total/phs' num2str(z) '.nii.gz']);
        save_nii(make_nii(phs_tissue_c,spatial_res),['/data/simu_POCSnt1/patch48/phs_tissue/phs' num2str(z) '.nii.gz']);
        save_nii(make_nii(qsm_c,spatial_res),['/data/simu_POCSnt1/patch48/qsm/qsm' num2str(z) '.nii.gz']);

        z=z+1;

    end
end