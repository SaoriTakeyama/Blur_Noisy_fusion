%%%% TV-based image restoration with blur and noisy image pair via ADMM %%%%

% Original Paper

% S. Takeyama, S. Ono, and I. Kumazawa, "Image restoration with multiple hard constraints 
% on data-fidelity to blurred/noisy image pair," IEICE Transactions on Information and Systems, 
% vol. E100-D, no.9, pp. 1953-1961, Sep. 2017.

% spatially-varying blur case

clear;
close all;
addpath subfunction

load RANDOMSTATES
% rng(rand_state,'v5uniform')
% rng(randn_state,'v5normal')
%================================
% old style
rand('state', rand_state);
randn('state', randn_state);
%================================

load non-uniform256_kernel_motion15to5_-90to-45.mat

%% observation generation %%
imname = 'img18.jpg'; % latent image
u_org_b = double(imread(imname))/255;
[height, width, ~] = size(u_org_b);
u_org = imcrop(u_org_b,[(width/2 - 128) (height/2 - 128) 255 255]);
[rows, cols, dim] = size(u_org); %read height and width

sigma_1 = 2/255;
sigma_2 = 16/255;% noise standard deviation (normalized)
gamma = 0.1; % stepsize of ADMM

% imposing spatially-varying blur
u1 = u_org(:,:,1);
u2 = u_org(:,:,2);
u3 = u_org(:,:,3);
v1 = Afull*u1(:);
v2 = Afull*u2(:);
v3 = Afull*u3(:);
v1 = reshape(v1,[256,256]);
v2 = reshape(v2,[256,256]);
v3 = reshape(v3,[256,256]);
v = cat(3, v1, v2, v3);

v_1 = v + sigma_1*randn(rows, cols, dim); % observation (blur+noise)
v_2 = u_org + sigma_2*randn(rows, cols, dim); %observation(noise)

psf = fspecial('motion', 9, -67); % motion blur psf
psfsize = size(psf);
blu = zeros(rows, cols);
blu(1:psfsize(1), 1:psfsize(2)) = psf; 
blu = circshift(blu, [-(psfsize(1)-1)/2 -(psfsize(2)-1)/2]); % blur kernel
bluf = fft2(blu); 
bluf = repmat(bluf, 1, 1, dim);
bluft = conj(bluf);
Phi = @(z) real(ifftn((fftn(z)).*bluf)); % blur operator
Phit = @(z) real(ifftn((fftn(z)).*bluft));

e1 = 0.95*sqrt(3*N*sigma_1^2);  %fidelity parameter
e2 = 0.95*sqrt(3*N*sigma_2^2);

%% setting %%

% difference operator
D = @(z) cat(4, z([2:rows, 1],:,:) - z, z(:,[2:cols, 1],:)-z);
Dt = @(z) z([rows,1:rows-1],:,:,1) - z(:,:,:,1)...
    + z(:,[cols,1:cols-1],:,2) - z(:,:,:,2);

% for inversion in update of u (F is 2DFFT, D is difference operator with circulant boundary, and DtD = -L)
LFinv = zeros(rows,cols,dim);
LFinv(1,1) = 4;
LFinv(1,2) = -1;
LFinv(2,1) = -1;
LFinv(rows,1) = -1;
LFinv(1,cols) = -1;
FLFinv = fftn(LFinv);
K = bluft.*bluf + FLFinv + 2*ones(rows,cols,dim);

% variables
u = v_1;
z1 = D(v_1);
z2 = v_1;
z3 = Phi(v_1);
z4 = v_1;
d1 = z1;
d2 = z2;
d3 = z3;
d4 = z4;


%% main loop%%
maxIter = 10000;
stopcri = 1e-4;
for i = 1:maxIter
    
    upre = u;
    % update of u
    rhs = Dt(z1 - d1) + z2 - d2 + Phit(z3 - d3) + z4 - d4;
    u = ifftn(fftn(rhs)./K); % inversion via diagonalization by 2DFFT ‘ÎŠp‰»‚É‚æ‚é”½“]
    
    % update of z
    % prox of mixed L1,2 norm
    z1 = D(u) + d1;
    z1 = ProxTVnorm(z1, gamma);
    % project onto dinamic range
    z2 = u + d2;
    z2(z2 > 1) = 1;
    z2(z2 < 0) = 0;
    % project onto l2-norm ball
    z3 = Phi(u) + d3;
    length3 = sqrt(sum(sum(sum((v_1 - z3).^2))));
    if length3 > e1
    z3 = v_1 + (z3 - v_1) * (e1 / length3);
    end
    z4 = u + d4;
    length4 = sqrt(sum(sum(sum((v_2 - z4).^2))));
    if length4 > e2
    z4 = v_2 + (z4 - v_2) * (e2 / length4);
    end
    
    % update of d
    d1 = d1 + D(u) - z1;
    d2 = d2 + u - z2;
    d3 = d3 + Phi(u) - z3;
    d4 = d4 + u - z4;
    
    % stopping condition
    psnr = EvalImgQuality(u, u_org, 'PSNR');
    res = u - upre;
    error = norm(res(:),2);
    disp(['i = ', num2str(i), ' PSNR = ', num2str(psnr,4), ' error = ', num2str(error,4)])
    if error < stopcri
        break;
    end
end
%% result plot

psnrInput = EvalImgQuality(u, u_org, 'PSNR');
disp(['Output PSNR = ', num2str(psnrInput)]);

plotsize = [2, 2];
ImgPlot(u_org, 'Original', 1, [plotsize,1]);
ImgPlot(u, 'Restored', 1, [plotsize,2]);
ImgPlot(v_1, 'Observation(blur+noise)', 1, [plotsize,3]);
ImgPlot(v_2, 'Observation(noise)', 1, [plotsize,4]);
