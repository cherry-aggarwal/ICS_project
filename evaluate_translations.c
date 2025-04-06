#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <locale.h>
#include <ctype.h>
#include <time.h>

#define EMBEDDING_DIM 300
#define MAX_WORDS 200000
#define MAX_WORD_LEN 100
#define MAX_LINE_LEN 4096
#define MAX_TEST_PAIRS 10000
#define TOP_K 5
#define MIN_SIMILARITY 0.6  // Minimum similarity score for valid translations

typedef struct {
    char word[MAX_WORD_LEN];
    float vector[EMBEDDING_DIM];
} Embedding;

typedef struct {
    char source[MAX_WORD_LEN];
    char target[MAX_WORD_LEN];
} TranslationPair;

Embedding en_embeddings[MAX_WORDS];
Embedding fr_embeddings[MAX_WORDS];
TranslationPair test_pairs[MAX_TEST_PAIRS];
int en_count = 0;
int fr_count = 0;
int test_count = 0;

void normalize_vector(float* vec) {
    float norm = 0.0;
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        norm += vec[i] * vec[i];
    }
    norm = sqrt(norm);
    if (norm < 1e-6) {
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            vec[i] = 1.0 / sqrt(EMBEDDING_DIM);
        }
        return;
    }
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        vec[i] /= norm;
    }
}

int load_embeddings(const char* filename, Embedding* embeddings) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        return 0;
    }

    char line[MAX_LINE_LEN];
    int count = 0;
    int dim = 0;

    if (fgets(line, sizeof(line), file)) {
        int n, d;
        if (sscanf(line, "%d %d", &n, &d) == 2) {
            dim = d;
            if (dim != EMBEDDING_DIM) {
                fprintf(stderr, "Warning: Embedding dimension mismatch (%d vs %d)\n", dim, EMBEDDING_DIM);
            }
        } else {
            fseek(file, 0, SEEK_SET);
        }
    }

    while (fgets(line, sizeof(line), file) && count < MAX_WORDS) {
        char* token = strtok(line, " ");
        if (!token) continue;

        char cleaned[MAX_WORD_LEN] = {0};
        int j = 0;
        for (int i = 0; token[i] && j < MAX_WORD_LEN-1; i++) {
            if (isalpha(token[i])) {
                cleaned[j++] = tolower(token[i]);
            }
        }
        if (j == 0) continue;

        strncpy(embeddings[count].word, cleaned, MAX_WORD_LEN - 1);
        embeddings[count].word[MAX_WORD_LEN - 1] = '\0';

        int i = 0;
        while (i < EMBEDDING_DIM && (token = strtok(NULL, " \t\n\r"))) {
            embeddings[count].vector[i++] = atof(token);
        }

        if (i == EMBEDDING_DIM) {
            normalize_vector(embeddings[count].vector);
            count++;
        } else {
            fprintf(stderr, "Warning: Invalid vector dimension for word '%s'\n", embeddings[count].word);
        }
    }

    fclose(file);
    return count;
}

float cosine_similarity(const float* a, const float* b) {
    float dot = 0.0;
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        dot += a[i] * b[i];
    }
    return dot;
}

void find_top_k(const float* source_vector, int k, int* top_indices, float* top_scores) {
    for (int i = 0; i < k; i++) {
        top_indices[i] = -1;
        top_scores[i] = -2.0f;
    }

    for (int i = 0; i < fr_count; i++) {
        float sim = cosine_similarity(source_vector, fr_embeddings[i].vector);
        
        for (int j = 0; j < k; j++) {
            if (sim > top_scores[j]) {
                for (int l = k-1; l > j; l--) {
                    top_indices[l] = top_indices[l-1];
                    top_scores[l] = top_scores[l-1];
                }
                top_indices[j] = i;
                top_scores[j] = sim;
                break;
            }
        }
    }
}

const char* translate_word(const char* english_word, int top_k, int* top_indices, float* top_scores) {
    char cleaned[MAX_WORD_LEN] = {0};
    int j = 0;
    for (int i = 0; english_word[i] && j < MAX_WORD_LEN-1; i++) {
        if (isalpha(english_word[i])) {
            cleaned[j++] = tolower(english_word[i]);
        }
    }
    if (j == 0) return "<unk>";

    float* source_vector = NULL;
    for (int i = 0; i < en_count; i++) {
        if (strcmp(en_embeddings[i].word, cleaned) == 0) {
            source_vector = en_embeddings[i].vector;
            break;
        }
    }
    if (!source_vector) return "<unk>";

    find_top_k(source_vector, top_k, top_indices, top_scores);
    
    // Find the best translation that isn't the same as input and meets similarity threshold
    for (int i = 0; i < top_k && top_indices[i] >= 0; i++) {
        if (strcmp(fr_embeddings[top_indices[i]].word, cleaned) != 0 && 
            top_scores[i] >= MIN_SIMILARITY) {
            return fr_embeddings[top_indices[i]].word;
        }
    }
    
    // If no suitable alternative found, return the best match with <same> marker
    if (top_indices[0] >= 0) {
        if (strcmp(fr_embeddings[top_indices[0]].word, cleaned) == 0) {
            return "<same>";
        }
        return fr_embeddings[top_indices[0]].word;
    }
    
    return "<unk>";
}

int load_test_pairs(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening test file: %s\n", filename);
        return 0;
    }

    char line[MAX_LINE_LEN];
    int count = 0;

    while (fgets(line, sizeof(line), file) && count < MAX_TEST_PAIRS) {
        char* en_word = strtok(line, "\t\n ");
        char* fr_word = strtok(NULL, "\t\n ");
        
        if (en_word && fr_word) {
            strncpy(test_pairs[count].source, en_word, MAX_WORD_LEN - 1);
            strncpy(test_pairs[count].target, fr_word, MAX_WORD_LEN - 1);
            test_pairs[count].source[MAX_WORD_LEN - 1] = '\0';
            test_pairs[count].target[MAX_WORD_LEN - 1] = '\0';
            count++;
        }
    }

    fclose(file);
    return count;
}

void evaluate_accuracy() {
    if (test_count == 0) {
        printf("No test pairs loaded for evaluation.\n");
        return;
    }

    int correct_top1 = 0;
    int correct_top5 = 0;
    int total_processed = 0;
    int top_indices[TOP_K];
    float top_scores[TOP_K];

    clock_t start = clock();

    for (int i = 0; i < test_count; i++) {
        const char* translation = translate_word(test_pairs[i].source, TOP_K, top_indices, top_scores);
        
        // Check top-1 accuracy (ignoring <same> results)
        if (strcmp(translation, "<same>") != 0 && 
            strcmp(translation, test_pairs[i].target) == 0) {
            correct_top1++;
        }
        
        // Check top-5 accuracy
        for (int j = 0; j < TOP_K && top_indices[j] >= 0; j++) {
            if (strcmp(fr_embeddings[top_indices[j]].word, test_pairs[i].target) == 0) {
                correct_top5++;
                break;
            }
        }
        
        total_processed++;
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\nEvaluation Results:\n");
    printf("Test pairs processed: %d\n", total_processed);
    printf("Top-1 Accuracy: %.2f%% (%d/%d)\n", 
           100.0 * correct_top1 / total_processed, correct_top1, total_processed);
    printf("Top-5 Accuracy: %.2f%% (%d/%d)\n", 
           100.0 * correct_top5 / total_processed, correct_top5, total_processed);
    printf("Processing time: %.2f seconds (%.0f pairs/sec)\n", 
           elapsed, total_processed / elapsed);
}

int main() {
    setlocale(LC_ALL, "en_US.UTF-8");

    printf("Loading English embeddings...\n");
    en_count = load_embeddings("wiki.en.align.vec", en_embeddings);
    printf("Loaded %d English words.\n", en_count);

    printf("Loading French embeddings...\n");
    fr_count = load_embeddings("wiki.fr.align.vec", fr_embeddings);
    printf("Loaded %d French words.\n", fr_count);

    if (en_count == 0 || fr_count == 0) {
        fprintf(stderr, "Failed to load embeddings.\n");
        return 1;
    }

    printf("Loading test pairs...\n");
    test_count = load_test_pairs("test_data.txt");
    if (test_count > 0) {
        printf("Loaded %d test pairs.\n", test_count);
        evaluate_accuracy();
    } else {
        printf("No test pairs loaded. Continuing without evaluation.\n");
    }

    printf("\nInteractive translation (type 'quit' to exit):\n");
    char input[MAX_WORD_LEN];
    int top_indices[TOP_K];
    float top_scores[TOP_K];

    while (1) {
        printf("Enter English word: ");
        if (!fgets(input, sizeof(input), stdin)) {
            break;
        }
        
        input[strcspn(input, "\n")] = 0;
        
        if (strcmp(input, "quit") == 0) {
            break;
        }
        
        const char* translation = translate_word(input, TOP_K, top_indices, top_scores);
        
        if (strcmp(translation, "<same>") == 0) {
            printf("No suitable translation found (best match was same as input)\n");
        } else {
            printf("Translation: %s\n", translation);
        }
        
        printf("Top %d candidates:\n", TOP_K);
        for (int i = 0; i < TOP_K && top_indices[i] >= 0; i++) {
            printf("%d. %s (score: %.4f)", i+1, fr_embeddings[top_indices[i]].word, top_scores[i]);
            if (strcmp(fr_embeddings[top_indices[i]].word, input) == 0) {
                printf(" <same as input>");
            }
            printf("\n");
        }
    }

    return 0;
}
