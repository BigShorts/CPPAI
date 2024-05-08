#ifndef SRC_FUNCLLM
#define SRC_FUNCLLM
#include <functional>
#include <sstream>
#include <vector>
#include <common/common.h>
#include <llama.h>

std::unordered_map<std::string, llama_token> tokenMap = {
    { "<unk>", 0 },
    { "<s>", 1 }, // BOS
    { "</s>", 2 }, // EOS
    { "<|content|>", 32000 },
    { "<|recipient|>", 32001 },
    { "<|from|>", 32002 },
    { "<|stop|>", 32003 }
};

struct Tool {
    std::string type;
    struct Function {
        std::string name;
        std::string description;
        struct Parameters {
            std::string type;
            struct Property {
                std::string type;
                std::string description;
            }; std::unordered_map<std::string, Property> properties;
            std::vector<std::string> required;
        } parameters;
    } function;
};

struct ChatMessage {
    std::string role;
    std::string content;
    std::string name;
    struct ToolCall { 
        struct { 
            std::string name;
            std::string arguments;
        } function;
    }; std::vector<ToolCall> toolCalls;
    std::unordered_map<std::string, std::string> metadata;
};

class FuncLLM {
private:
    llama_context* ctx;
    llama_model* model;
    llama_batch batch;
    std::vector<Tool> tools;
    std::vector<ChatMessage> messages;
    std::vector<llama_token> tokensList;
    std::vector<std::string> errors;
    const int batchSize = 512;
    int nCtx;

    void tokenize(std::string message) {
        auto tokens = ::llama_tokenize(ctx, message, true);
        tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());
    }
    
    void tokenizeFunctionary() {
        tokensList = {};

        for (auto message : messages) {
            if (message.role == "user" || message.role == "system") {
                tokensList.push_back(tokenMap["<|from|>"]);
                tokenize(message.role + '\n');
                tokensList.push_back(tokenMap["<|recipient|>"]);
                tokenize("all\n");
                tokensList.push_back(tokenMap["<|content|>"]);
                tokenize(message.content + '\n');
            } else if (message.role == "tool") {
                tokensList.push_back(tokenMap["<|from|>"]);
                tokenize(message.name + '\n');
                tokensList.push_back(tokenMap["<|recipient|>"]);
                tokenize("all\n");
                tokensList.push_back(tokenMap["<|content|>"]);
                tokenize(message.content + '\n');
            } else {
                bool containContent = false;
            
                if (message.content != "") {
                    tokensList.push_back(tokenMap["<|from|>"]);
                    tokenize("assistant\n");
                    tokensList.push_back(tokenMap["<|recipient|>"]);
                    tokenize("all\n");
                    tokensList.push_back(tokenMap["<|content|>"]);
                    tokenize(message.content);

                    containContent = true;
                }

                if (message.toolCalls.size()) {
                    for (int i = 0; i < message.toolCalls.size(); i++) {
                        if (!(i == 0 && !containContent)) {
                            tokenize("\n");
                        }

                        tokensList.push_back(tokenMap["<|from|>"]);
                        tokenize("assistant\n");
                        tokensList.push_back(tokenMap["<|recipient|>"]);
                        tokenize(message.toolCalls[i].function.name + '\n');
                        tokensList.push_back(tokenMap["<|content|>"]);
                        tokenize(message.toolCalls[i].function.arguments);
                    }
                }

                tokensList.push_back(tokenMap["<|stop|>"]);
                tokenize("\n");
            }
        }

        tokensList.push_back(tokenMap["<|from|>"]);
        tokenize("assistant\n");
        tokensList.push_back(tokenMap["<|recipient|>"]);
        
        // Print tokensList as string
        std::cout << "\n\nTokensList:\n";
        for (auto token : tokensList) {
            std::cout << llama_token_to_piece(ctx, token);
        }
        std::cout << "\n\n";
    }

    void decodeTokens(std::vector<llama_token> tokens) {
        int batchesNeeded = std::ceil(float(tokens.size()) / float(batchSize));
        for (int i = 0; i < batchesNeeded; i++) {
            llama_batch_clear(batch);
            int jUpper = (tokens.size()-i*batchSize < batchSize) ? tokens.size()-i*batchSize : batchSize;
            for (size_t j = 0; j < jUpper; j++) {
                llama_batch_add(batch, tokens[j+i*batchSize], j+i*batchSize, { 0 }, false);
            }

            if (i == batchesNeeded-1) batch.logits[batch.n_tokens - 1] = true;
            llama_decode(ctx, batch);
        }
    }

    void decodeMessage(std::string message) {
        tokensList = ::llama_tokenize(ctx, message, true);
        decodeTokens(tokensList);
    }

public:
    FuncLLM(std::string modelPath, int seed, int nCtx, int nThreads, int nThreadsBatch, std::string systemMessage = "", int nGpuLayers = -1, bool numa = false) {
        this->nCtx = nCtx;
        llama_backend_init(numa);

        llama_model_params modelParams = llama_model_default_params();
        modelParams.n_gpu_layers = nGpuLayers;

        model = llama_load_model_from_file(modelPath.c_str(), modelParams);

        if (model == nullptr)
            errors.push_back("Unable to load model at path " + modelPath);

        llama_context_params ctxParams = llama_context_default_params();

        ctxParams.seed = seed;
        ctxParams.n_ctx = nCtx;
        ctxParams.n_threads = nThreads;

        ctxParams.n_threads_batch = nThreadsBatch == -1 ? nThreads : nThreadsBatch;

        ctx = llama_new_context_with_model(model, ctxParams);

        if (ctx == nullptr)
            errors.push_back("Unable to initialize context");

        batch = llama_batch_init(batchSize, 0, 1);

        const auto t_main_start = ggml_time_us();

        if (systemMessage != "") {
            messages.push_back(ChatMessage{ "system", systemMessage });
        }
    }

    ~FuncLLM() {
        llama_batch_free(batch);

        llama_free(ctx);
        llama_free_model(model);
    }

    std::vector<std::string> getErrors() {
        return errors;
    }

    std::string response(std::string prompt, int nLen, bool live = false) {
        llama_reset_timings(ctx);
        messages.push_back(ChatMessage{ "user", prompt });

        if (nLen > nCtx)
            return "Error: nLen > nCtx";

        llama_kv_cache_clear(ctx);

        tokenizeFunctionary();
        decodeTokens(tokensList);

        std::stringstream content;
        int nCur = tokensList.size();
        int tokensGenerated = 0;
        bool contentStarted = false;
        std::string tag;

        ChatMessage newMessage;
        newMessage.role = "assistant";

        while (tokensGenerated < nLen) {
            auto nVocab = llama_n_vocab(model);
            auto* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(nVocab);

            for (llama_token tokenId = 0; tokenId < nVocab; tokenId++)
                candidates.emplace_back(llama_token_data{ tokenId, logits[tokenId], 0.0f });

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            const llama_token newTokenId = llama_sample_token(ctx, &candidates_p);

            if (contentStarted) {
                if (newTokenId == tokenMap["<|stop|>"]) {
                    contentStarted = false;
                    break;
                } else {
                    content << llama_token_to_piece(ctx, newTokenId);

                    if (live) {
                        std::cout << llama_token_to_piece(ctx, newTokenId);
                        std::fflush(stdout);
                    }
                }
            }

            if (newTokenId == tokenMap["<|content|>"]) {
                contentStarted = true;
            }

            llama_batch_clear(batch);
            llama_batch_add(batch, newTokenId, nCur, { 0 }, true);
            if (llama_decode(ctx, batch)) {
                return "Error: llama_decode() failed";
            }

            tokensGenerated++;
            nCur++;
        }

        messages.push_back(ChatMessage{ "assistant", content.str() });

        return messages.back().content;
    }

    void addTool(Tool tool) {
        tools.push_back(tool);
    }

    void printTimings() {
        llama_print_timings(ctx);
    }

    bool ok() {
        return errors.size() == 0;
    }
};

#endif /* SRC_FUNCLLM */
