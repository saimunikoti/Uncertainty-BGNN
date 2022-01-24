clc
clear all
close all

% [V,X] = datageneration_ieee37_matpower();
measurement_data = load('s.mat');
V = measurement_data.V;
n= 111; % nodes
M = 48; %time 

[s,t,G,A,Laplacian_orig] = graph_related_code();
cmr_alist = 0.1*[5]; % percentage of known entries in adjacency matrix

%%
[Phi_A] = Phifunctions1(M,n, cmr_alist);
ASamp = Phi_A.*A;
A2 = estimate_adjacency(n,M,abs(V), Phi_A, ASamp);

%%
A2temp=A2;
A2temp(A2temp < 0.249) = 0;

Gestimated = graph(A2temp);
Estimated_edges2 = Gestimated.Edges;
presentedges = ismember(G.Edges,Estimated_edges2(:,1));
missing_edges = length(find(presentedges == 0));

notfalseedges = ismember(Estimated_edges2(:,1),G.Edges);
false_edges = length(find(notfalseedges == 0));

ER= 100*(missing_edges + false_edges)/108;

Accuracy1=  100*length(find(presentedges == 1))/108;
% figure(2)
% spy(abs(A2temp));
% xlabel('nodes'); ylabel('nodes'); title('Estimated Adjacency matrix');
%