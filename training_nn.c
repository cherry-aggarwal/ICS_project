#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "load_embeddings.h"


#define VOCAB_SIZE 10
#define EMBED_DIM 10
#define LEARNING_RATE 0.01
#define EPOCHS 10000

// Function to compute Mean Squared Error
float compute_loss(float y_true[EMBED_DIM], float y_pred[EMBED_DIM]) {
    float loss = 0;
    for (int i = 0; i < EMBED_DIM; i++) {
        float diff = y_true[i] - y_pred[i];
        loss += diff * diff;
    }
    return loss / EMBED_DIM;
}

// Training function using gradient descent
void train_nn(float en_embeddings[VOCAB_SIZE][EMBED_DIM], float fr_embeddings[VOCAB_SIZE][EMBED_DIM], weights[EMBED_DIM][EMBED_DIM]) {
    float weights[EMBED_DIM][EMBED_DIM]; // Transformation matrix
    for (int i = 0; i < EMBED_DIM; i++)
        for (int j = 0; j < EMBED_DIM; j++)
            weights[i][j] = ((float) rand() / RAND_MAX) * 2 - 1; // Random init

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0;
        
        for (int i = 0; i < VOCAB_SIZE; i++) {
            float y_pred[EMBED_DIM] = {0};

            // Forward pass: y_pred = en_embedding * weights
            for (int j = 0; j < EMBED_DIM; j++) {
                for (int k = 0; k < EMBED_DIM; k++) {
                    y_pred[j] += en_embeddings[i][k] * weights[k][j];
                }
            }

            // Compute loss
            total_loss += compute_loss(fr_embeddings[i], y_pred);

            // Backpropagation: Adjust weights
            for (int j = 0; j < EMBED_DIM; j++) {
                for (int k = 0; k < EMBED_DIM; k++) {
                    float gradient = (y_pred[j] - fr_embeddings[i][j]) * en_embeddings[i][k];
                    weights[k][j] -= LEARNING_RATE * gradient;
                }
            }
        }

        if (epoch % 1000 == 0)
            printf("Epoch %d, Loss: %.4f\n", epoch, total_loss / VOCAB_SIZE);
    }

    printf("Training complete!\n");
}

float cosine_similarity(float vec1[EMBED_DIM], float vec2[EMBED_DIM]) {
    float dot_product = 0, norm1 = 0, norm2 = 0;
    for (int i = 0; i < EMBED_DIM; i++) {
        dot_product += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }
    return dot_product / (sqrt(norm1) * sqrt(norm2));
}

// Function to perform inference
void translate_word(float en_embedding[EMBED_DIM], float fr_embeddings[VOCAB_SIZE][EMBED_DIM], float weights[EMBED_DIM][EMBED_DIM]) {
    float translated[EMBED_DIM] = {0};

    // Transform English embedding to predicted French embedding
    for (int j = 0; j < EMBED_DIM; j++) {
        for (int k = 0; k < EMBED_DIM; k++) {
            translated[j] += en_embedding[k] * weights[k][j];
        }
    }

    // Find closest French word in embedding space
    int best_match = 0;
    float best_similarity = -1;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        float similarity = cosine_similarity(translated, fr_embeddings[i]);
        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_match = i;
        }
    }

    printf("Predicted French word is at index: %d with similarity: %.4f\n", best_match, best_similarity);
}


// Main function
int main() {
    float en_embeddings[VOCAB_SIZE][EMBED_DIM];
    float fr_embeddings[VOCAB_SIZE][EMBED_DIM];
    float weights[EMBED_DIM][EMBED_DIM];

    load_embeddings("word2vec_en.csv", en_embeddings);
    load_embeddings("word2vec_fr.csv", fr_embeddings);

    train_nn(en_embeddings, fr_embeddings, weights);

    FILE *file = fopen("trained_weights.bin", "wb");
    fwrite(weights, sizeof(float), EMBED_DIM * EMBED_DIM, file);
    fclose(file);

    
    return 0;
}
