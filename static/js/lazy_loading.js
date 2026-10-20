// CineSense - Lazy Loading Manager
// Handles lazy loading functionality and cache monitoring

const LazyLoadingManager = {
    // Configuration
    config: {
        useLazyLoading: true,
        strategy: 'mixed', // mixed, genre, popularity, exploration
        cacheRefreshInterval: 30000, // 30 seconds
        enableCacheMonitoring: true
    },
    
    // Cache statistics
    cacheStats: {
        movieCache: null,
        vectorCache: null,
        lastUpdate: null
    },
    
    // Initialize lazy loading
    async initialize() {
        // wave5 fix (J1.7): index.html inline init + this DOMContentLoaded
        // init run on the same page — guard against double initialization
        // (double intervals, duplicate requests).
        if (this._initialized) {
            return;
        }
        this._initialized = true;

        console.log('🚀 Lazy Loading Manager initialized');

        
        // Start cache monitoring if enabled
        if (this.config.enableCacheMonitoring) {
            this.startCacheMonitoring();
        }
        
        // Update cache stats immediately
        await this.updateCacheStats();
    },
    
    // Get comparison pair (uses lazy loading if enabled)
    async getComparisonPair() {
        try {
            if (this.config.useLazyLoading) {
                const response = await window.CineSense.API.getComparisonPairLazy();
                
                // Update cache stats
                if (response.cache_stats) {
                    this.cacheStats = response.cache_stats;
                    this.cacheStats.lastUpdate = new Date();
                }
                
                return {
                    movie1: response.movie1,
                    movie2: response.movie2,
                    lazyLoaded: response.lazy_loaded,
                    cacheStats: response.cache_stats
                };
            } else {
                const response = await window.CineSense.API.getComparisonPair();
                return {
                    movie1: response.movie1,
                    movie2: response.movie2,
                    lazyLoaded: false
                };
            }
        } catch (error) {
            console.error('Error getting comparison pair:', error);
            throw error;
        }
    },
    
    // Get recommendations (uses lazy loading if enabled)
    async getRecommendations(limit = 20, strategy = null) {
        try {
            const useStrategy = strategy || this.config.strategy;
            
            if (this.config.useLazyLoading) {
                const response = await window.CineSense.API.getRecommendationsLazy(limit, useStrategy);
                
                // Update cache stats
                if (response.cache_stats) {
                    this.cacheStats = response.cache_stats;
                    this.cacheStats.lastUpdate = new Date();
                }
                
                return {
                    movies: response.movies,
                    personalized: response.personalized,
                    lazyLoaded: response.lazy_loaded,
                    strategy: response.strategy,
                    cacheStats: response.cache_stats
                };
            } else {
                const response = await window.CineSense.API.getRecommendations(limit);
                return {
                    movies: response.movies,
                    personalized: response.personalized,
                    lazyLoaded: false
                };
            }
        } catch (error) {
            console.error('Error getting recommendations:', error);
            throw error;
        }
    },
    
    // Update cache statistics
    async updateCacheStats() {
        try {
            const stats = await window.CineSense.API.getCacheStats();
            if (!stats || !stats.cache_manager) {
                // wave5 fix (J1.13): an error/null response must not overwrite
                // the stats with an error-shaped object — keep it null so
                // callers render zeros instead of fake numbers.
                this.cacheStats = null;
                throw new Error('cache stats unavailable');
            }
            this.cacheStats = stats.cache_manager;

            this.cacheStats.lastUpdate = new Date();
            
            // Emit event for UI updates
            window.dispatchEvent(new CustomEvent('cacheStatsUpdated', { 
                detail: this.cacheStats 
            }));
            
            return this.cacheStats;
        } catch (error) {
            console.error('Error updating cache stats:', error);
            return null;
        }
    },
    
    // Get detailed cache monitoring data
    async getCacheMonitor() {
        try {
            // wave5 fix (J1.8): with CacheMonitorWidget deleted, the
            // 'cacheMonitorUpdated' event had no listeners — dispatch removed.
            return await window.CineSense.API.monitorCache();
        } catch (error) {
            console.error('Error getting cache monitor:', error);
            return null;
        }
    },
    
    // Start automatic cache monitoring
    startCacheMonitoring() {
        console.log('📊 Cache monitoring started');

        // wave5 fix (J1.7): clear any previous interval first so repeated
        // initialize() calls don't stack multiple 30s pollers.
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
        }

        // Update every 30 seconds
        this.monitoringInterval = setInterval(async () => {
            await this.updateCacheStats();
        }, this.config.cacheRefreshInterval);
    },
    
    // Stop cache monitoring
    stopCacheMonitoring() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            console.log('📊 Cache monitoring stopped');
        }
    },
    
    // Toggle lazy loading
    toggleLazyLoading(enabled) {
        this.config.useLazyLoading = enabled;
        console.log(`Lazy loading ${enabled ? 'enabled' : 'disabled'}`);
        
        // Store preference
        localStorage.setItem('lazyLoadingEnabled', enabled);
    },
    
    // Change candidate generation strategy
    setStrategy(strategy) {
        const validStrategies = ['mixed', 'genre', 'popularity', 'exploration'];
        
        if (validStrategies.includes(strategy)) {
            this.config.strategy = strategy;
            console.log(`Strategy changed to: ${strategy}`);
            
            // Store preference
            localStorage.setItem('candidateStrategy', strategy);
        } else {
            console.warn(`Invalid strategy: ${strategy}`);
        }
    },
    
    // Get cache health status
    getCacheHealth() {
        if (!this.cacheStats || !this.cacheStats.lastUpdate) {
            return 'unknown';
        }
        
        const movieCount = this.cacheStats.movie_count || 0;
        const vectorCount = this.cacheStats.vector_count || 0;
        
        // Check if cache is healthy (> 30% full)
        const movieUsage = (movieCount / 100) * 100;
        const vectorUsage = (vectorCount / 500) * 100;
        
        if (movieUsage < 30 || vectorUsage < 30) {
            return 'low';
        } else if (movieUsage > 90 || vectorUsage > 90) {
            return 'full';
        } else {
            return 'healthy';
        }
    },
    
    // Format cache stats for display
    formatCacheStats() {
        if (!this.cacheStats) {
            return 'Cache stats unavailable';
        }
        
        const movieCount = this.cacheStats.movie_count || 0;
        const vectorCount = this.cacheStats.vector_count || 0;
        const movieHits = this.cacheStats.movie_hits || 0;
        const movieMisses = this.cacheStats.movie_misses || 0;
        const vectorHits = this.cacheStats.vector_hits || 0;
        const vectorMisses = this.cacheStats.vector_misses || 0;
        
        const movieTotal = movieHits + movieMisses;
        const vectorTotal = vectorHits + vectorMisses;
        
        const movieHitRate = movieTotal > 0 ? (movieHits / movieTotal * 100).toFixed(2) : 0;
        const vectorHitRate = vectorTotal > 0 ? (vectorHits / vectorTotal * 100).toFixed(2) : 0;
        
        return {
            movieCache: `${movieCount}/100 (${movieHitRate}% hit rate)`,
            vectorCache: `${vectorCount}/500 (${vectorHitRate}% hit rate)`,
            health: this.getCacheHealth()
        };
    }
};

// wave5 fix (J1.8): the entire CacheMonitorWidget class was deleted — nothing
// instantiated it and its only event listener fed off 'cacheMonitorUpdated',
// an event the /monitor page's own dashboard does not use (dead feature).

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Load saved preferences
    const lazyLoadingEnabled = localStorage.getItem('lazyLoadingEnabled');
    if (lazyLoadingEnabled !== null) {
        LazyLoadingManager.config.useLazyLoading = lazyLoadingEnabled === 'true';
    }
    
    const savedStrategy = localStorage.getItem('candidateStrategy');
    if (savedStrategy) {
        // wave5 fix (J1.12): a corrupt saved value used to be applied verbatim
        // and made every lazy call 400. Validate against the backend whitelist
        // (routes.py VALID_STRATEGIES) and fall back to 'mixed'.
        const VALID_STRATEGIES = ['mixed', 'genre', 'popularity', 'exploration'];
        if (VALID_STRATEGIES.includes(savedStrategy)) {
            LazyLoadingManager.config.strategy = savedStrategy;
        } else {
            console.warn(`Invalid saved strategy "${savedStrategy}" — resetting to 'mixed'`);
            localStorage.removeItem('candidateStrategy');
            LazyLoadingManager.config.strategy = 'mixed';
        }
    }
    
    // Initialize lazy loading
    LazyLoadingManager.initialize();
    
    console.log('🎬 CineSense Lazy Loading initialized');
    console.log('Strategy:', LazyLoadingManager.config.strategy);
    console.log('Lazy Loading:', LazyLoadingManager.config.useLazyLoading ? 'Enabled' : 'Disabled');
});

// Export for global access
window.LazyLoadingManager = LazyLoadingManager;
