function [s,t,G,A,Laplacian_orig] = graph_related_code()

s = [1 2 3 4 5 6 7   8 9 10  11 12 10 11 12 13 14 15 16 17 18 16 17 18 25 26 27 28 29 30 31 32 33 31 32 33 37 38 39 40 41 42 40 41 42 ...
     43 44 45 43 44 45 52 53 54 55 56 57 58 59 60 58 59 60 28 29 30 28 29 30  7  8 9 73  74 75 73 74 75  7  8  9 82 83 84 85 86 87 88 89 90 ...
     85 86 87 94 95 96 97  98   99 94 95 96 103 104 105 103 104 105];
 
t = [4 5 6 7 8 9 10 11 12 13 14 15 25 26 27 16 17 18 19 20 21 22 23 24 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 52 53 54 ...
     46 47 48 49 50 51 55 56 57 58 59 60 61 62 63 64 65 66 70 71 72 67 68 69 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 ... 
     94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111];
 
%t([8 7]) = t([7 8]);
%t([36 35]) = t([35 36]);
%[s,t] = swap_elements(s,t);

G = graph(s,t,'omitselfloops');
% plot(G,'EdgeLabel',G.Edges.Weight);
A = full(adjacency(G));
% imagesc(A);
% axis on;
% xticks(1:size(A, 2));
% yticks(1:size(A, 1));
% colorbar
% D = degree(G);
Laplacian_orig = full(laplacian(G));
% nL = (diag(D)^(-.5))*Laplacian_orig*(diag(D)^(-.5));
end