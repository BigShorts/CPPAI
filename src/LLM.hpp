#ifndef SRC_LLM
#define SRC_LLM
#include <vector>
#include <sstream>
#include <llama.h>
#include <common/common.h>

class LLM {
private:
    llama_batch batch;
    llama_model* model;
    llama_context* ctx;
    std::vector<std::string> errors;
    std::vector<llama_token> tokensList;
    int nCtx;

public:
    LLM(std::string modelPath, int seed, int nCtx, int nThreads, int nThreadsBatch, int nGpuLayers = -1, bool numa = false) {
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

        int n_cur    = batch.n_tokens;
        int n_decode = 0;

        const auto t_main_start = ggml_time_us();
    }

    ~LLM() {
        llama_batch_free(batch);

        llama_free(ctx);
        llama_free_model(model);
    }

    std::vector<std::string> getErrors() {
        return errors;
    }

    std::string response(std::string prompt, int nLen, bool echo = false) {
        if (nLen == 0)
            return "";

        if (nLen > nCtx)
            return "Error: nLen > nCtx";

        batch = llama_batch_init(512, 0, 1);
        llama_kv_cache_clear(ctx);
        tokensList = ::llama_tokenize(ctx, prompt, true);
        for (size_t i = 0; i < tokensList.size(); i++) {
            llama_batch_add(batch, tokensList[i], i, { 0 }, false);
        }

        std::stringstream response;
        if (echo)
            for (auto id : tokensList)
                response << llama_token_to_piece(ctx, id);

        batch.logits[batch.n_tokens - 1] = true;
        if (llama_decode(ctx, batch) != 0)
            return "Error: llama_decode() failed";

        int nCur = batch.n_tokens;
        int nDecode = 0;

        while (nCur < nLen) {
            auto  nVocab = llama_n_vocab(model);
            auto* logits  = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(nVocab);

            for (llama_token tokenId = 0; tokenId < nVocab; tokenId++)
                candidates.emplace_back(llama_token_data{ tokenId, logits[tokenId], 0.0f });

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            const llama_token newTokenId = llama_sample_token_greedy(ctx, &candidates_p);

            if (newTokenId == llama_token_eos(model) || nCur == nLen) {
                std::cout << "\n";
                break;
            }

            response << llama_token_to_piece(ctx, newTokenId);

            llama_batch_clear(batch);

            llama_batch_add(batch, newTokenId, nCur, { 0 }, true);

            nDecode += 1;
            nCur += 1;

            if (llama_decode(ctx, batch))
                return "Error: llama_decode() failed";
        }

        return response.str();
    }

    void printTimings() {
        llama_print_timings(ctx);
        llama_reset_timings(ctx);
    }

    bool ok() {
        return errors.size() == 0;
    }
};

#endif /* SRC_LLM */
