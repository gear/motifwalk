# Index file for embedding results

## Blogcatalog3 dataset

### Id: BC3001

1. epoch = 1; 
2. emb\_dim = 128; 
3. neg\_samp = 15; 
4. num\_skip = 5; 
5. num\_walk = 10;
6. walk\_length = 80;
7. nodes\_per\_epoch = 200;
8. batch\_size = 512;
9. skip\_window = 10;
10. Method: Random walk;
11. Negative sampling: Unifromly random (distort = 0);

NOTE: batch\_size is the number of samples per loss update in keras.


### Id: BC3002

1. epoch = 1; 
2. emb\_dim = 128; 
3. neg\_samp = 15; 
4. num\_skip = 5; 
5. num\_walk = 10;
6. walk\_length = 80;
7. nodes\_per\_epoch = 200;
8. batch\_size = 256;
9. skip\_window = 10;
10. Method: Random walk;
11. Negative sampling: Unifromly random (distort = 0);


### Id: BC3003

1. epoch = 1; 
2. emb\_dim = 128; 
3. neg\_samp = 5; 
4. num\_skip = 15; 
5. num\_walk = 10;
6. walk\_length = 80;
7. nodes\_per\_epoch = 200;
8. batch\_size = 1024;
9. skip\_window = 10;
10. Method: Random walk;
11. Negative sampling: Unifromly random (distort = 0);


### Id: BC3004

1. epoch = 1; 
2. emb\_dim = 128; 
3. neg\_samp = 5; 
4. num\_skip = 15; 
5. num\_walk = 10;
6. walk\_length = 80;
7. nodes\_per\_epoch = 200;
8. batch\_size = 64;
9. skip\_window = 10;
10. Method: Random walk;
11. Negative sampling: Unifromly random (distort = 0);


### Id: BC3005

1. epoch = 1; 
2. emb\_dim = 128; 
3. neg\_samp = 5; 
4. num\_skip = 15; 
5. num\_walk = 1;
6. walk\_length = 40;
7. nodes\_per\_epoch = 200;
8. batch\_size = 1024;
9. skip\_window = 10;
10. Method: Motif walk (old implementation);
11. Negative sampling: Unigram by degree;


### Id: BC3001

1. epoch = 1; 
2. emb\_dim = 128; 
3. neg\_samp = 5; 
4. num\_skip = 15; 
5. num\_walk = 10;
6. walk\_length = 80;
7. nodes\_per\_epoch = 200;
8. batch\_size = 512;
9. skip\_window = 10;
10. Method: Random walk;
11. Negative sampling: Unigram by degree;

