%Clear all
clear; close all; clc;
%Turn off warnings
warning('off');
%load in rgb and label images
theFiles = dir('./train/imgs/*.jpg');
for k = 1: length(theFiles)
    %Read in files
    baseFileName = theFiles(k).name;
    pat = digitsPattern(3);

    alphanumerics = extract(baseFileName,pat);
    class = alphanumerics(1);
    
        
    class = char(class);
    if class == char('017')
        mkdir(class)
    end
    if class == char('017')
        mkdir(class)
    end
    if class == char('017')
        mkdir(class)
    end
    if class == char('017')
        mkdir(class)
    end
    if class == char('017')
        mkdir(class)
    end
    if class == char('017')
        mkdir(class)
    end
    if class == char('017')
        mkdir(class)
    end
    if class == char('017')
        mkdir(class)
    end
    if class == char('055')
        mkdir(class)
    end
    if class == char('071')
        mkdir(class)
    end
    if class == char('072')
        mkdir(class)
    end
    if class == char('099')
        mkdir(class)
    end
    if class == char('102')
        mkdir(class)
    end
   
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
        %read file
    imageArrayOrig = imread(fullFileName);
    mkdir(class)
    %analysis = fullfile(class, 'edge_detection');
    %mkdir(analysis); 
    %analysis_canny = fullfile(analysis, 'canny');
    %mkdir(analysis_canny);
    %analysis_sobel = fullfile(analysis, 'sobel');
    %mkdir(analysis_sobel);
    %analysis_fourier = fullfile(class, 'fourier');
    %mkdir(analysis_fourier);
    
    %imwrite(imageArrayOrig,fullfile(class, imageArrayOrig))
    %Spatial and Spectral Analysis
    %Greyscale images
    %Edge detection images
    Igray = rgb2gray(imageArrayOrig);
    cannyedge= edge(Igray, 'canny');
    %imshow(cannyedge);
    sobeledge= edge(Igray, 'sobel');
    %imshow(sobeledge);
    
    % Get the dimensions of the image.  https://uk.mathworks.com/matlabcentral/answers/46699-how-to-segment-divide-an-image-into-4-equal-halves
    %[rows, columns, ~] = size(imageArrayOrig);
    % Get the rows and columns to split at,
    % Taking care to handle odd-size dimensions:
    %col1 = 1;
    %col2 = floor(columns/2);
    %col3 = col2 + 1;
    %row1 = 1;
    %row2 = floor(rows/2);
    %row3 = row2 + 1;
    % Now crop
    %upperLeft = imcrop(imageArrayOrig, [col1 row1 col2 row2]);
    %upperRight = imcrop(imageArrayOrig, [col3 row1 columns - col2 row2]);
    %lowerLeft = imcrop(imageArrayOrig, [col1 row3 col2 row2]);
    %lowerRight = imcrop(imageArrayOrig, [col3 row3 columns - col2 rows - row2]);
    
    
    %Get values for all images
    %[SR_neg_orig, ST_neg_orig, ST_pos_orig] = fourier_collect(imageArrayOrig);
    %[SR_neg_UL, ST_neg_UL, ST_pos_UL] = fourier_collect(upperLeft);
    %[SR_neg_UR, ST_neg_UR, ST_pos_UR] = fourier_collect(upperRight);
    %[SR_neg_LL, ST_neg_LL, ST_pos_LL] = fourier_collect(lowerLeft);
    %[SR_neg_LR, ST_neg_LR, ST_pos_LR] = fourier_collect(lowerRight);
    
    %plot all together
    %figure_1 = plotter(SR_neg_orig, SR_neg_UL, SR_neg_UR, SR_neg_LL, SR_neg_LR, 'Radius 0 to -pi');
    %figure_2 = plotter(ST_neg_orig, ST_neg_UL, ST_neg_UR, ST_neg_LL, ST_neg_LR, 'Radius 0 to -pi');
    %figure_3 = plotter(ST_pos_orig, ST_pos_UL, ST_pos_UR, ST_pos_LL, ST_pos_LR, 'Radius 0 to pi');
    
    
    
    %imwrite(imageArrayOrig,fullfile(class,baseFileName));
    %imwrite(cannyedge,fullfile(analysis_canny,baseFileName));
    %imwrite(sobeledge,fullfile(analysis_sobel,baseFileName));
    
    %baseFileName = baseFileName(1:end-3);
    %baseFileName_1 = append(baseFileName, char('figure1.jpg'));
    %baseFileName_2 = append(baseFileName, char('figure2.jpg'));
    %baseFileName_3 = append(baseFileName, char('figure3.jpg'));
    %saveas(figure_1,fullfile(analysis_fourier,baseFileName_1));
    %saveas(figure_2,fullfile(analysis_fourier,baseFileName_2));
    %saveas(figure_3,fullfile(analysis_fourier,baseFileName_3));
end


function [fig] = plotter(orig, UL, UR, LL, LR, xlab)
fig = figure;
hold on
plot(log(orig));
plot(log(UL));
plot(log(UR));
plot(log(LL));
plot(log(LR));
hold off
xlabel(xlab);
ylabel('Frequency')
legend('Full', 'Upper Left', 'Upper Right','Lower Left','Lower Right');
end

function[SR_neg, ST_neg, ST_pos] = fourier_collect(image)

%imshow(image)
%Convert to gray
Igray = rgb2gray(image);
%Reszie the image to 256,256
resized = imresize(Igray, [256 256]);
%2D fast fourier transform
f = fft2(resized, 256, 256);
%Shift to centre
f = fftshift(f);
%Show new image
%figure;
%imshow(f);
%Show the colormap of the transformed image
%figure;
transformed_img = log(1+abs(f));
%imshow(transformed_img,[]); colormap(jet); colorbar;

%log the image for analysis
transformed_img_log = log(transformed_img);
%S(R) array for values SR pos is same as neg
SR_neg = [];
%S(theta) array for values
ST_neg = [];
ST_pos = [];

%distance for circle implementation will make pi
distance = 180;
%half of 256 -1 error otherwise
half_img = 127;

%For the radius r (half of image since it will go to edge)
for r = 0:half_img
    %set S_R to 0
    S_R = 0;
    %for theta between 0 and -pi
    for theta = 0:-1:-distance
        %degrees to radians each value from 0 to -pi
        theta = deg2rad(theta);
        %transform the polar coordinates from polar to cartesian
        [x, y] = pol2cart(theta, r);
        %round the value add the other half of the image
        x = round(x + 128);
        y = round(y + 128);
        %S_R is S_R + the location of x and y in the log of the fourier
        %transform image
        S_R = S_R + transformed_img_log(x, y);
    end
    %SR to plot is SR and S_R location
    SR_neg = [SR_neg, S_R];
end

%Theta for loop 0 to -pi
for theta = 0:-1:-distance
    %degrees to radians
    theta = deg2rad(theta);
    %set S_T to 0
    S_T = 0;
    %For the radius r (half of image since it will go to edge)
    for r = 0:half_img
        %transform the polar coordinates from polar to cartesian
        [x,y] = pol2cart(theta, r);
        %round the value add the other half of the image
        x = round(x + 128);
        y = round(y + 128);
        %S_T is S_T + the location of x and y in the log of the fourier
        %transform image
        S_T = S_T + transformed_img_log(x, y);
    end
    %ST to plot is ST and S_T location
    ST_neg = [ST_neg, S_T];
end

%Theta for loop 0 to pi
for theta = 0:1:distance
    %degrees to radians
    theta = deg2rad(theta);
    %set S_T to 0
    S_T = 0;
    %For the radius r (half of image since it will go to edge)
    for r = 0:half_img
        %transform the polar coordinates from polar to cartesian
        [x,y] = pol2cart(theta, r);
        %round the value add the other half of the image
        x = round(x + 128);
        y = round(y + 128);
        %S_T is S_T + the location of x and y in the log of the fourier
        %transform image
        S_T = S_T + transformed_img_log(x, y);
    end
    %ST to plot is ST and S_T location
    ST_pos = [ST_pos, S_T];
end

end