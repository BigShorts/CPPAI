#include "Articles.hpp"

Articles::Articles(std::string dbPath) {
    char *zErrMsg = 0;
    int rc = sqlite3_open(dbPath.c_str(), &db);

    if (rc) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
    }

    sqlite3_stmt *stmt;
    rc = sqlite3_prepare_v2(db, "SELECT id FROM Articles ORDER BY id DESC LIMIT 1", -1, &stmt, 0);

    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare statement: %s\n", sqlite3_errmsg(db));
        exit(1);
    }

    rc = sqlite3_step(stmt);

    if (rc == SQLITE_ROW) {
        totalArticles = sqlite3_column_int(stmt, 0);
    }

    sqlite3_finalize(stmt);
}

Articles::~Articles() {
    sqlite3_close(db);
}

// Table:
// CREATE TABLE "Articles" (
//  "id"	        INTEGER NOT NULL,
//  "title"	        TEXT,
//  "link"	        TEXT,
//  "textContent"	TEXT,
//  "summary"       TEXT,
//   PRIMARY KEY(""id"" AUTOINCREMENT)
// );

Article Articles::getNextArticle() {
    Article article;
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, "SELECT * FROM Articles WHERE summary IS NULL LIMIT 1", -1, &stmt, 0);

    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare statement: %s\n", sqlite3_errmsg(db));
        return { -1, "", "", "" };
    }

    rc = sqlite3_step(stmt);

    if (rc == SQLITE_ROW) {
        article.id = sqlite3_column_int(stmt, 0);
        article.title = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
        article.link = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2)));
        article.textContent = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3)));
    } else {
        article.id = -1;
    }

    sqlite3_finalize(stmt);
    return article;
}

Article Articles::getNextArticlePortion(int portion, int totalPortions) {
    Article article;
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, "SELECT * FROM Articles WHERE summary IS NULL AND id > ? ORDER BY id ASC LIMIT 1", -1, &stmt, 0);

    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare statement: %s\n", sqlite3_errmsg(db));
        return { -1, "", "", "" };
    }

    int portionalRange = (float)totalArticles / (float)totalPortions;
    int startID = portionalRange * (float)portion;

    rc = sqlite3_bind_int(stmt, 1, startID);
    rc = sqlite3_step(stmt);

    if (rc == SQLITE_ROW) {
        article.id = sqlite3_column_int(stmt, 0);
        article.title = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
        article.link = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2)));
        article.textContent = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3)));
    } else {
        article.id = -1;
    }

    if (article.id > portionalRange + (portion * portionalRange)) {
        article.id = -1;
    }

    sqlite3_finalize(stmt);
    return article;
}

void Articles::updateSummary(int id, std::string summary) {
    sqlite3_stmt *stmt;
    int rc = sqlite3_prepare_v2(db, "UPDATE Articles SET summary = ? WHERE id = ?", -1, &stmt, 0);

    if (rc != SQLITE_OK) {
        fprintf(stderr, "Failed to prepare statement: %s\n", sqlite3_errmsg(db));
        return;
    }

    rc = sqlite3_bind_text(stmt, 1, summary.c_str(), -1, SQLITE_STATIC);
    rc = sqlite3_bind_int(stmt, 2, id);
    rc = sqlite3_step(stmt);

    if (rc != SQLITE_DONE) {
        fprintf(stderr, "Failed to update summary: %s\n", sqlite3_errmsg(db));
    }

    sqlite3_finalize(stmt);
}
