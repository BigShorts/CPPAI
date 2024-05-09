#ifndef SRC_ARTICLES
#define SRC_ARTICLES
#include <string>
#include <sqlite3.h>

struct Article {
    int id;
    std::string title;
    std::string link;
    std::string textContent;
};

class Articles {
private:
    sqlite3* db;
    int totalArticles = 0;

public:
    Articles(std::string dbPath);
    ~Articles();

    Article getNextArticle();
    Article getNextArticlePortion(int portion, int totalPortions);

    void updateSummary(int id, std::string summary);
};

#endif /* SRC_ARTICLES */
