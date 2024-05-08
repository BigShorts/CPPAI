#ifndef SRC_LLM
#define SRC_LLM
#include <vector>
#include <sstream>
#include <llama.h>
#include <common/common.h>

enum class LLMType {
    SUMMARISE,
    CHAT
};

struct ChatMessage {
    std::string role;
    std::string content;
};

class LLM {
protected:
    LLMType type;
    llama_context* ctx;
    llama_model* model;
    llama_batch batch;
    std::vector<llama_token> tokensList;
    std::vector<ChatMessage> messages;
    std::vector<std::string> errors;
    std::string systemMessage = "";
    const int batchSize = 512;
    int nCtx;

    virtual void tokenize() = 0;

    void decodeTokens(std::vector<llama_token> tokens);
    void decodeMessage(std::string message);

    void genMessagesSummarise(std::string prompt);
    void genMessagesChat(std::string prompt);

public:
    LLM(std::string modelPath, int seed, int nCtx, int nThreads, int nThreadsBatch, LLMType type, std::string systemMessage = "", int nGpuLayers = -1, bool numa = false);
    ~LLM();

    std::vector<std::string> getErrors();

    std::string response(std::string prompt, int nLen, bool live = false);

    void printTimings();
    bool ok();
};

#endif /* SRC_LLM */
