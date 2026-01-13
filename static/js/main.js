// CineSense - Main JavaScript

// Utility Functions
const Utils = {
    // Format runtime (minutes to hours)
    formatRuntime: (minutes) => {
        if (!minutes) return 'N/A';
        const hours = Math.floor(minutes / 60);
        const mins = minutes % 60;
        return `${hours}h ${mins}m`;
    },
    
    // Format number with commas
    formatNumber: (num) => {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    },
    
    // Get poster URL
    getPosterUrl: (path, size = 'w500') => {
        if (!path) return 'https://via.placeholder.com/500x750?text=No+Poster';
        return `https://image.tmdb.org/t/p/${size}${path}`;
    },
    
    // Get backdrop URL
    getBackdropUrl: (path, size = 'w1280') => {
        if (!path) return 'https://via.placeholder.com/1280x720?text=No+Backdrop';
        return `https://image.tmdb.org/t/p/${size}${path}`;
    },
    
    // Show toast notification
    showToast: (message, type = 'success') => {
        const toast = document.createElement('div');
        toast.className = `fixed bottom-4 right-4 px-6 py-4 rounded-lg shadow-lg z-50 ${
            type === 'success' ? 'bg-green-600' : 'bg-red-600'
        }`;
        toast.textContent = message;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
};

// API Client
const API = {
    baseUrl: '/api',
    
    async request(endpoint, options = {}) {
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'API request failed');
            }
            
            return data;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },
    
    // User endpoints
    async login(username, password) {
        return this.request('/user/login', {
            method: 'POST',
            body: JSON.stringify({ username, password })
        });
    },
    
    async signup(username, email, password) {
        return this.request('/user/signup', {
            method: 'POST',
            body: JSON.stringify({ username, email, password })
        });
    },
    
    async logout() {
        return this.request('/user/logout', { method: 'POST' });
    },
    
    async getProfile() {
        return this.request('/user/profile');
    },
    
    // Movie endpoints
    async getRecommendations(limit = 20) {
        return this.request(`/recommendations?limit=${limit}`);
    },
    
    async getFeatured() {
        return this.request('/featured');
    },
    
    async getComparisonPair() {
        return this.request('/compare');
    },
    
    async submitFeedback(movie1Id, movie2Id, chosenId) {
        return this.request('/feedback', {
            method: 'POST',
            body: JSON.stringify({
                movie1_id: movie1Id,
                movie2_id: movie2Id,
                chosen_id: chosenId
            })
        });
    },
    
    async getMovieDetail(movieId) {
        return this.request(`/movie/${movieId}`);
    },
    
    async searchMovies(query, limit = 20) {
        return this.request(`/movie/search?q=${encodeURIComponent(query)}&limit=${limit}`);
    },
    
    async getTopRated(orderBy = 'elo_score', limit = 20) {
        return this.request(`/movie/top-rated?order_by=${orderBy}&limit=${limit}`);
    },
    
    async getMoviesByGenre(genre, limit = 20) {
        return this.request(`/movie/by-genre/${encodeURIComponent(genre)}?limit=${limit}`);
    }
};

// Movie Card Component
function createMovieCard(movie) {
    const posterUrl = Utils.getPosterUrl(movie.poster_path);
    const rating = movie.tmdb_rating ? movie.tmdb_rating.toFixed(1) : 'N/A';
    
    return `
        <div class="movie-card cursor-pointer" onclick="window.location.href='/movie/${movie.movie_id}'">
            <div class="relative rounded-lg overflow-hidden aspect-[2/3] group">
                <img src="${posterUrl}" alt="${movie.title}" class="w-full h-full object-cover">
                
                <div class="absolute inset-0 bg-gradient-to-t from-black via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                    <div class="absolute bottom-0 left-0 right-0 p-4">
                        <h3 class="font-bold text-lg mb-2 line-clamp-2">${movie.title}</h3>
                        <div class="flex items-center justify-between text-sm">
                            <span class="flex items-center">
                                <svg class="w-4 h-4 text-yellow-400 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"></path>
                                </svg>
                                ${rating}
                            </span>
                            <span class="text-gray-400">${movie.release_year || 'N/A'}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    console.log('🎬 CineSense initialized');
});

// Export for use in other scripts
window.CineSense = {
    Utils,
    API,
    createMovieCard
};
