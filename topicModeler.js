import { kmeans } from 'ml-kmeans';

export class TopicModeler {
    constructor() {
        // Basic English stopwords
        this.stopWords = new Set([
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", 
            "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", 
            "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", 
            "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", 
            "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", 
            "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", 
            "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
            "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", 
            "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", 
            "under", "until", "up", "very", 
            "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", 
            "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
        ]);
    }

    /**
     * Main pipeline method
     * @param {string[]} texts - Original text documents
     * @param {number[][]} embeddings - Embedding vectors
     * @param {number} k - Number of clusters (optional, default auto-estimated or 5)
     */
    run(texts, embeddings, k = null) {
        if (!texts || texts.length === 0) return [];

        // 1. Cluster
        // If k is not provided, use a simple heuristic: sqrt(N/2) or just 5 if small
        const numClusters = k || Math.max(2, Math.floor(Math.sqrt(texts.length / 2)));
        const { clusters } = this._cluster(embeddings, numClusters);

        // 2. Group docs by cluster
        const clusteredDocs = new Map();
        for (let i = 0; i < clusters.length; i++) {
            const label = clusters[i];
            if (!clusteredDocs.has(label)) {
                clusteredDocs.set(label, { docs: [], indices: [] });
            }
            clusteredDocs.get(label).docs.push(texts[i]);
            clusteredDocs.get(label).indices.push(i);
        }

        // 3. c-TF-IDF & Keyword Extraction
        const results = [];
        
        // Calculate class-based vocab stats
        // total_clusters is essentially clusteredDocs.size
        const totalClusters = clusteredDocs.size;
        
        // Pre-process all clusters to get word counts per cluster
        const clusterWordCounts = new Map(); // label -> Map<word, count>
        
        for (const [label, data] of clusteredDocs.entries()) {
            const wordCounts = this._countWords(data.docs);
            clusterWordCounts.set(label, wordCounts);
        }

        // Calculate IDF for each word across clusters: 
        // IDF(t) = log(1 + A / F) where A = avg number of words per class? 
        // Or user provided formula: log(1 + (total_clusters / freq_in_clusters))
        // freq_in_clusters: number of clusters containing the word? Or total frequency?
        // Standard c-TF-IDF usually defines IDF as: log(1 + TotalDocuments / FrequencyAcrossAllDocuments) ??? 
        // The user prompted: "log(1 + (total_clusters / freq_in_clusters))"
        // Let's interpret 'freq_in_clusters' as "number of clusters containing the word t".
        
        // Build map: word -> set of clusters containing it
        const wordInClusters = new Map();
        for (const [label, wordCounts] of clusterWordCounts.entries()) {
            for (const word of wordCounts.keys()) {
                if (!wordInClusters.has(word)) {
                    wordInClusters.set(word, new Set());
                }
                wordInClusters.get(word).add(label);
            }
        }

        // Generate output
        for (const [label, data] of clusteredDocs.entries()) {
            const wordCounts = clusterWordCounts.get(label);
            const totalWordsInCluster = Array.from(wordCounts.values()).reduce((a, b) => a + b, 0);

            // Compute scores
            const scores = [];
            for (const [word, count] of wordCounts.entries()) {
                // TF = frequency in this cluster (can be raw count or normalized)
                // c-TF-IDF usually uses raw count * IDF, or (count/total) * IDF
                // We'll use count for "frequency of words in that cluster" as per prompt prompt.
                // But normalization is better for ranking. Let's use count.
                const tf = count; 
                
                // IDF
                const clustersWithWord = wordInClusters.get(word).size;
                const idf = Math.log(1 + (totalClusters / clustersWithWord));
                
                const score = tf * idf;
                scores.push({ word, score });
            }

            // Sort by score desc
            scores.sort((a, b) => b.score - a.score);
            
            // Top keywords
            const keywords = scores.slice(0, 5).map(s => s.word);
            
            // Generate label
            // e.g. "word1_word2_word3"
            const topicLabel = keywords.slice(0, 3).join("_") || `Topic ${label}`;

            results.push({
                id: label,
                label: topicLabel,
                keywords: keywords,
                docs: data.docs,
                indices: data.indices, // Include original indices for mapping back
                // We don't necessarily need to return all embeddings in the JSON object unless asked,
                // but the prompt asked for "embeddings": [ ...array of vectors... ]
                embeddings: data.indices.map(i => embeddings[i]) 
            });
        }

        return results.sort((a, b) => a.id - b.id);
    }

    _cluster(embeddings, k) {
        // ml-kmeans expects array of arrays
        const result = kmeans(embeddings, k, { initialization: 'kmeans++' });
        return { clusters: result.clusters };
    }

    _countWords(docs) {
        const counts = new Map();
        for (const doc of docs) {
            // Normalize: lower case, remove punctuation
            const words = doc.toLowerCase()
                .replace(/[^\w\s]/g, ' ') // remove non-alpha chars (keep spaces)
                .split(/\s+/)
                .filter(w => w.length > 2 && !this.stopWords.has(w)); // filter short and stopwords

            for (const w of words) {
                counts.set(w, (counts.get(w) || 0) + 1);
            }
        }
        return counts;
    }
}
