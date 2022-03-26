package com.search.deeplearning.lucene;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.StopAnalyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.miscellaneous.PerFieldAnalyzerWrapper;
import org.apache.lucene.analysis.synonym.SynonymGraphFilter;
import org.apache.lucene.analysis.synonym.SynonymMap;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.*;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.CharsRef;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class SynonymExpandSearch {

    public static void main(String[] args) throws IOException, ParseException {
        SynonymMap.Builder builder = new SynonymMap.Builder();
        builder.add(new CharsRef("aeroplane"), new CharsRef("plane"), true);
        final SynonymMap map = builder.build();

        Analyzer indexTimeAnalyzer = new Analyzer() {
            @Override
            protected TokenStreamComponents createComponents(String fieldName) {
                Tokenizer tokenizer = new WhitespaceTokenizer();
                SynonymGraphFilter synFilter = new SynonymGraphFilter(tokenizer, map, true);
                return new TokenStreamComponents(tokenizer, synFilter);
            }
        };

        Map<String, Analyzer> perFieldAnalyzers = new HashMap<>();
        CharArraySet stopwords = new CharArraySet(Arrays.asList("a", "an", "the"), true);
        perFieldAnalyzers.put("pages", new StopAnalyzer(stopwords));
        perFieldAnalyzers.put("title", new WhitespaceAnalyzer());
        Analyzer analyzer = new PerFieldAnalyzerWrapper(indexTimeAnalyzer, perFieldAnalyzers);
        Analyzer searchTimeAnalyzer = new WhitespaceAnalyzer();

        Path path = Paths.get("./data/lucene/luceneidx");
        Directory directory = FSDirectory.open(path);

        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        IndexWriter writer = new IndexWriter(directory, config);

        /*Document dl4s = new Document();
        dl4s.add(new TextField("title", "Deep Learning for search", Field.Store.YES));
        dl4s.add(new TextField("page", "Living in the information age ...", Field.Store.YES));

        Document rs = new Document();
        rs.add(new TextField("title", "Relevant search", Field.Store.YES));
        rs.add(new TextField("page", "Getting a search engine to behave ...", Field.Store.YES));

        writer.addDocument(dl4s);
        writer.addDocument(rs);*/

        Document aeroplaneDoc = new Document();
        aeroplaneDoc.add(new TextField("title","Aeroplane", Field.Store.YES));
        aeroplaneDoc.add(new TextField("author","Red Hot Chili Peppers", Field.Store.YES));
        aeroplaneDoc.add(new TextField("year","1995", Field.Store.YES));
        aeroplaneDoc.add(new TextField("album","One Hot Minute", Field.Store.YES));
        aeroplaneDoc.add(new TextField("text","I like pleasure spiked with pain and music is my aeroplane ...", Field.Store.YES));

        writer.addDocument(aeroplaneDoc);
        writer.commit();
        writer.close();

        IndexReader reader = DirectoryReader.open(directory);
        QueryParser parser = new QueryParser("text", searchTimeAnalyzer);
        Query query = parser.parse("plane");

        IndexSearcher searcher = new IndexSearcher(reader);
        TopDocs hits = searcher.search(query, 10);
        for (int i = 0; i < hits.scoreDocs.length; i++) {
            ScoreDoc scoreDoc = hits.scoreDocs[i];
            Document doc = reader.document(scoreDoc.doc);
            System.out.println(doc.get("title") + " by " + doc.get("author"));
        }
    }
}
