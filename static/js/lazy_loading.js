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
            const monitor = await window.CineSense.API.monitorCache();
            
            // Emit event for dashboard updates
            window.dispatchEvent(new CustomEvent('cacheMonitorUpdated', { 
                detail: monitor 
            }));
            
            return monitor;
        } catch (error) {
            console.error('Error getting cache monitor:', error);
            return null;
        }
    },
    
    // Start automatic cache monitoring
    startCacheMonitoring() {
        console.log('📊 Cache monitoring started');
        
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

// Cache Monitoring Widget
class CacheMonitorWidget {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.isVisible = false;
        
        if (this.container) {
            this.render();
            this.attachEventListeners();
        }
    }
    
    render() {
        this.container.innerHTML = `
            <div class="cache-monitor-widget" style="position: fixed; bottom: 20px; right: 20px; background: rgba(0,0,0,0.9); color: white; padding: 15px; border-radius: 10px; min-width: 300px; display: none; z-index: 1000;">
                <div class="flex justify-between items-center mb-3">
                    <h3 class="text-lg font-bold">Cache Monitor</h3>
                    <button class="close-monitor text-gray-400 hover:text-white">&times;</button>
                </div>
                
                <div class="cache-stats">
                    <div class="stat-item mb-2">
                        <div class="flex justify-between text-sm">
                            <span>Movie Cache:</span>
                            <span class="movie-cache-value">-/-</span>
                        </div>
                        <div class="progress-bar bg-gray-700 h-2 rounded mt-1">
                            <div class="movie-cache-bar bg-blue-500 h-2 rounded" style="width: 0%"></div>
                        </div>
                    </div>
                    
                    <div class="stat-item mb-2">
                        <div class="flex justify-between text-sm">
                            <span>Vector Cache:</span>
                            <span class="vector-cache-value">-/-</span>
                        </div>
                        <div class="progress-bar bg-gray-700 h-2 rounded mt-1">
                            <div class="vector-cache-bar bg-green-500 h-2 rounded" style="width: 0%"></div>
                        </div>
                    </div>
                    
                    <div class="stat-item mb-2">
                        <div class="flex justify-between text-sm">
                            <span>Movie Hit Rate:</span>
                            <span class="movie-hit-rate">-%</span>
                        </div>
                    </div>
                    
                    <div class="stat-item mb-2">
                        <div class="flex justify-between text-sm">
                            <span>Vector Hit Rate:</span>
                            <span class="vector-hit-rate">-%</span>
                        </div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="flex justify-between text-sm">
                            <span>Status:</span>
                            <span class="cache-status badge">Loading...</span>
                        </div>
                    </div>
                </div>
                
                <div class="text-xs text-gray-400 mt-3">
                    Last updated: <span class="last-update">Never</span>
                </div>
            </div>
        `;
    }
    
    attachEventListeners() {
        const widget = this.container.querySelector('.cache-monitor-widget');
        const closeBtn = this.container.querySelector('.close-monitor');
        
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.hide());
        }
        
        // Listen for cache updates
        window.addEventListener('cacheMonitorUpdated', (event) => {
            this.updateDisplay(event.detail);
        });
    }
    
    updateDisplay(data) {
        if (!data) return;
        
        const widget = this.container.querySelector('.cache-monitor-widget');
        if (!widget) return;
        
        // Update movie cache
        const movieCount = data.movie_cache?.count || 0;
        const movieMax = data.movie_cache?.max_size || 100;
        const movieUsage = data.movie_cache?.usage_percent || 0;
        const movieHitRate = data.movie_cache?.hit_rate || 0;
        
        widget.querySelector('.movie-cache-value').textContent = `${movieCount}/${movieMax}`;
        widget.querySelector('.movie-cache-bar').style.width = `${movieUsage}%`;
        widget.querySelector('.movie-hit-rate').textContent = `${movieHitRate}%`;
        
        // Update vector cache
        const vectorCount = data.vector_cache?.count || 0;
        const vectorMax = data.vector_cache?.max_size || 500;
        const vectorUsage = data.vector_cache?.usage_percent || 0;
        const vectorHitRate = data.vector_cache?.hit_rate || 0;
        
        widget.querySelector('.vector-cache-value').textContent = `${vectorCount}/${vectorMax}`;
        widget.querySelector('.vector-cache-bar').style.width = `${vectorUsage}%`;
        widget.querySelector('.vector-hit-rate').textContent = `${vectorHitRate}%`;
        
        // Update status
        const health = data.status?.health || 'unknown';
        const statusBadge = widget.querySelector('.cache-status');
        statusBadge.textContent = health;
        statusBadge.className = 'cache-status badge';
        
        if (health === 'healthy') {
            statusBadge.classList.add('bg-green-600');
        } else if (health === 'low_hit_rate') {
            statusBadge.classList.add('bg-yellow-600');
        } else {
            statusBadge.classList.add('bg-gray-600');
        }
        
        // Update timestamp
        widget.querySelector('.last-update').textContent = new Date().toLocaleTimeString();
    }
    
    show() {
        const widget = this.container.querySelector('.cache-monitor-widget');
        if (widget) {
            widget.style.display = 'block';
            this.isVisible = true;
            
            // Fetch initial data
            LazyLoadingManager.getCacheMonitor();
        }
    }
    
    hide() {
        const widget = this.container.querySelector('.cache-monitor-widget');
        if (widget) {
            widget.style.display = 'none';
            this.isVisible = false;
        }
    }
    
    toggle() {
        if (this.isVisible) {
            this.hide();
        } else {
            this.show();
        }
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Load saved preferences
    const lazyLoadingEnabled = localStorage.getItem('lazyLoadingEnabled');
    if (lazyLoadingEnabled !== null) {
        LazyLoadingManager.config.useLazyLoading = lazyLoadingEnabled === 'true';
    }
    
    const savedStrategy = localStorage.getItem('candidateStrategy');
    if (savedStrategy) {
        LazyLoadingManager.config.strategy = savedStrategy;
    }
    
    // Initialize lazy loading
    LazyLoadingManager.initialize();
    
    console.log('🎬 CineSense Lazy Loading initialized');
    console.log('Strategy:', LazyLoadingManager.config.strategy);
    console.log('Lazy Loading:', LazyLoadingManager.config.useLazyLoading ? 'Enabled' : 'Disabled');
});

// Export for global access
window.LazyLoadingManager = LazyLoadingManager;
window.CacheMonitorWidget = CacheMonitorWidget;
