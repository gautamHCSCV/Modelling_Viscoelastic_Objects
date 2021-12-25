
k = 1;
cl = 22;
sf = 2;
 
    load ('O1_raw.mat')
    figure(k)
    for phi = 1:13
        time = 1/1000:1/1000:length(raw_pm_r{1,22}{1,phi}{1,2})/1000;
        plot(time,raw_fm_r{1,cl}{1,phi}{1,sf})
        xlabel('time')
        ylabel('signals for obj1')
        hold on
    end
    hold off
   

%%
k = k+1;
    for phi = 1:13
        fk = raw_fm_r{1,22}{1,phi}{1,2};
        gk = fft(fk);
        figure(k)
        g0k = (gk(1));
        plot(phi,g0k,'--gs',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5])
        xlabel('k')
        ylabel('gok')
        hold on
    end
    hold off
    k = k+1;
    for phi = 1:13
        fk = raw_fm_r{1,22}{1,phi}{1,2};
        gk = fft(fk);
        figure(k)
        g1k = 2*real(gk(2))/length(fk);
        plot(phi,g1k,'--gs',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5])
        xlabel('k')
        ylabel('g1k')
        hold on
    end
    hold off
    k = k+1;
    for phi = 1:13
        fk = raw_fm_r{1,22}{1,phi}{1,2};
        gk = fft(fk);
        figure(k)
        g2k = -2*imag(gk(2))/length(fk);
        plot(phi,g2k,'--gs',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5])
        xlabel('k')
        ylabel('g2k')
        hold on
    end
    hold off
    k = k+1;
    for phi = 1:13
        fk = raw_fm_r{1,22}{1,phi}{1,2};
        gk = fft(fk);
        figure(k)
        g3k = 2*real(gk(3))/length(fk);
        plot(phi,g3k,'--gs',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5])
        xlabel('k')
        ylabel('g3k')
        hold on
    end
    hold off
    k = k+1;
    for phi = 1:13
        fk = raw_fm_r{1,22}{1,phi}{1,2};
        gk = fft(fk);
        figure(k)
        g4k = -2*imag(gk(3))/length(fk);
        plot(phi,g4k,'--gs',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5])
        xlabel('k')
        ylabel('g4k')
        hold on
    end
    hold off
    k = k+1;
    for phi = 1:13
        fk = raw_fm_r{1,22}{1,phi}{1,2};
        gk = fft(fk);
        figure(k)
        g5k = 2*real(gk(4))/length(fk);
        plot(phi,g5k,'--gs',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5])
        xlabel('k')
        ylabel('g5k')
        hold on
    end
    hold off
    k = k+1;
    for phi = 1:13
        fk = raw_fm_r{1,22}{1,phi}{1,2};
        gk = fft(fk);
        figure(k)
        g10k = -2*imag(gk(6))/length(fk);
        plot(phi,g10k,'--gs',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5])
        xlabel('k')
        ylabel('g10k')
        hold on
   
    end
   
   