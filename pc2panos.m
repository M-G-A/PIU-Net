%% define paths to dataset
sets = ['2019-03-16_15.48.59'; ...
        '2019-03-24_17.42.18'];
path = ['/data.nas/staff/michael/',sets(2,:),'/'];
ptCloud = pcread([path,'pointcloud.ply']);
lines = readmatrix([path,'pano/pano-poses.csv']); % contains rotation (in quaternion) and translation of captured ground truth panoramas

%% setup new directory for the results
test = 'test_name';
mkdir([path,'rendered_panos',test]);
mkdir([path,'rendered_panos_depth',test]);

%% scale values to [0,1]
R=double(ptCloud.Color(:,1))./255;
B=double(ptCloud.Color(:,2))./255;
G=double(ptCloud.Color(:,3))./255;

%% Rotation and translation of the train-set
T = lines(:,4:6);
Q = lines(:,7:10);

%% render
parpool('local',20);
parfor p = 1:length(T)

    t=-T(p,:);
    q=Q(p,:);
    
    % create transformation matrix
    RT = quatrotate(q,ptCloud.Location+t);
    RT = (rotz(90) * RT')';
    x = RT(:,1);
    y = RT(:,2);
    z = RT(:,3);

    % calculate spherical presentation of the points
    [a,e,r] = cart2sph(x,y,z);

    % create image grid (size = 2^12 x 2^13)
    x1=2^13-uencode(double(a),13,pi);
    x2=uencode(double(e),12,pi/2);

    % sort points according to distance
    [rs,I]=sort(r,'descend');
    I2 = fliplr(I);
    
    % sets how many pixels should be coloured around the projection
    interp = 6; %rgb
    interp2 = 20; %depth
    
    pic = zeros(2^12+2*interp,2^13+2*interp,3);
    pic2 = zeros(2^12+2*interp2,2^13+2*interp2,3);
    for i = 1:length(I)
        n=2*interp+1;
        n2=2*interp2+1;
        %% rgb
        rgb=reshape(repelem([R(I(i)),B(I(i)),G(I(i))],n*n),n,n,3);
        pic(2^12-x2(I(i)):2^12-x2(I(i))+n-1,x1(I(i)):x1(I(i))+n-1,:)=rgb;
        %% depth
        rgb2=reshape(repelem([0,0,r(I2(i))/13],n2*n2),n2,n2,3);
        pic2(2^12-x2(I2(i)):2^12-x2(I2(i))+n2-1,x1(I2(i)):x1(I2(i))+n2-1,:)=rgb2;
    end
    
    % write
    imwrite(pic(interp+1:end-interp,interp+1:end-interp,:),[path,'rendered_panos',test,'/', num2str(p-1,'%05.f'),'-pano.jpg'])
    imwrite(pic2(interp2+1:end-interp2,interp2+1:end-interp2,:),[path,'rendered_panos_depth',test,'/', num2str(p-1,'%05.f'),'-pano_depth.jpg'])
    imshow(pic(interp+1:end-interp,interp+1:end-interp,:))
end


