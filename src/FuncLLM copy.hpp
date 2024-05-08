#ifndef SRC_FUNCLLM_20COPY
#define SRC_FUNCLLM_20COPY
#ifndef SRC_FUNCLLM
#define SRC_FUNCLLM
#include <functional>
#include <sstream>
#include <vector>
#include <common/common.h>
#include <llama.h>

#define CONTENT 32000
#define RECIPIENT 32001
#define FROM 32002
#define STOP 32003

struct Tool {
    std::string type;
    struct Function {
        std::string name;
        std::string description;
        struct Parameters {
            std::string type;
            struct Property {
                std::string name;
                std::string arguments;
            };
            std::vector<Property> properties;
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
     } toolCall;
    std::vector<ToolCall> toolCalls;
};

class FuncLLM {
private:
    llama_context* ctx;
    llama_model* model;
    llama_batch batch;
    std::unordered_map<std::string, Tool> tools;
    std::vector<ChatMessage> messages;
    std::vector<llama_token> tokensList;
    std::vector<std::string> errors;
    const int batchSize = 512;
    int nCtx;

    
    
    void tokenizeFunctionary() {
        tokensList = {};
        for (auto message : messages) {
            if (message.role == "user" || message.role == "system") {
                tokensList.push_back(FROM);
                auto tokens = ::llama_tokenize(ctx, message.role + '\n', true);
                tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());
                tokensList.push_back(RECIPIENT);
                tokens = ::llama_tokenize(ctx, "all\n", true);
                tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());
                tokensList.push_back(CONTENT);
                tokens = ::llama_tokenize(ctx, message.content + '\n', true);
                tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());
            } else if (message.role == "tool") {
                tokensList.push_back(FROM);
                auto tokens = ::llama_tokenize(ctx, message.name + '\n', true);
                tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());
                tokensList.push_back(RECIPIENT);
                tokens = ::llama_tokenize(ctx, "all\n", true);
                tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());
                tokensList.push_back(CONTENT);
                tokens = ::llama_tokenize(ctx, message.content + '\n', true);
                tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());
            } else {
                bool containContent = false;
                if (message.content != "") {
                    tokensList.push_back(FROM);
                    auto tokens = ::llama_tokenize(ctx, "assistant\n", true);
                    tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());
                    tokensList.push_back(RECIPIENT);
                    tokens = ::llama_tokenize(ctx, "all\n", true);
                    tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());
                    tokensList.push_back(CONTENT);
                    tokens = ::llama_tokenize(ctx, message.content, true);
                    tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());
                    containContent = true;
                }

                if (message.toolCalls.size() > 0) {
                    for (auto toolCall : message.toolCalls) {
                        tokensList.push_back(FROM);
                        auto tokens = ::llama_tokenize(ctx, "assistant\n", true);
                        tokens.push_back(RECIPIENT);
                        auto newTokens = ::llama_tokenize(ctx, toolCall.function.name + '\n', true);
                        tokens.insert(tokens.end(), newTokens.begin(), newTokens.end());
                        tokens.push_back(CONTENT);
                        newTokens = ::llama_tokenize(ctx, toolCall.function.arguments + '\n', true);
                        tokens.insert(tokens.end(), newTokens.begin(), newTokens.end());

                        if (containContent) {
                            tokensList.push_back(CONTENT);
                            newTokens = ::llama_tokenize(ctx, "\n", true);
                            tokensList.insert(tokensList.end(), newTokens.begin(), newTokens.end());
                        }

                        tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());
                    }
                }

                tokensList.push_back(STOP);
            }
        }

        tokensList.push_back(FROM);
        auto tokens = ::llama_tokenize(ctx, "assistant\n", true);
        tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());

        tokensList.push_back(RECIPIENT);
        tokens = ::llama_tokenize(ctx, "all\n", true);
        tokensList.insert(tokensList.end(), tokens.begin(), tokens.end());

        tokensList.push_back(CONTENT);
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

        std::stringstream response;
        int nCur = tokensList.size();
        bool tagFound = false;
        std::string tag;

        while (nCur < nLen) {
            auto nVocab = llama_n_vocab(model);
            auto* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(nVocab);

            for (llama_token tokenId = 0; tokenId < nVocab; tokenId++)
                candidates.emplace_back(llama_token_data{ tokenId, logits[tokenId], 0.0f });

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            const llama_token newTokenId = llama_sample_token(ctx, &candidates_p);

            if (newTokenId == STOP || nCur == nLen) {
                llama_batch_clear(batch);
                llama_batch_add(batch, newTokenId, nCur, { 0 }, true);
                llama_decode(ctx, batch);

                nCur += 1;
                break;
            }
            
            response << llama_token_to_piece(ctx, newTokenId);

            if (live) {
                std::cout << llama_token_to_piece(ctx, newTokenId);
                std::fflush(stdout);
            }

            llama_batch_clear(batch);
            llama_batch_add(batch, newTokenId, nCur, { 0 }, true);
            if (llama_decode(ctx, batch))
                return "Error: llama_decode() failed";

            nCur += 1;
        }

        messages.push_back(ChatMessage{ "assistant", response.str() });

        return messages.back().content;
    }

    void addTool(std::string name, Tool tool) {
        tools[name] = tool;
    }

    void printTimings() {
        llama_print_timings(ctx);
    }

    bool ok() {
        return errors.size() == 0;
    }
};

#endif /* SRC_FUNCLLM */


#endif /* SRC_FUNCLLM_20COPY */
