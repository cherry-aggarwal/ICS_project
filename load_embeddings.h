#ifndef LOAD_EMBEDDINGS_C
#define LOAD_EMBEDDINGS_C
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB_SIZE 10   // Adjust as needed
#define EMBED_DIM 10    // Same as in Python

// Function to load embeddings from CSV
void load_embeddings(const char *filename, float embeddings[VOCAB_SIZE][EMBED_DIM]) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }

    char line[256];
    int row = 0;
    while (fgets(line, sizeof(line), file) && row < VOCAB_SIZE) {
        char *token = strtok(line, ",");
        int col = 0;
        while (token && col < EMBED_DIM) {
            embeddings[row][col] = atof(token);
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }
    fclose(file);
}

// Function to print embeddings
void print_embeddings(float embeddings[VOCAB_SIZE][EMBED_DIM]) {
    for (int i = 0; i < VOCAB_SIZE; i++) {
        printf("Word %d: ", i);
        for (int j = 0; j < EMBED_DIM; j++) {
            printf("%.3f ", embeddings[i][j]);
        }
        printf("\n");
    }
}


#endif