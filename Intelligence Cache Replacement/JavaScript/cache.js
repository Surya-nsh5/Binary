document.addEventListener('DOMContentLoaded', () => {

    // --- DOM Elements ---
    const requestBtn = document.getElementById('request-btn');
    const pageInput = document.getElementById('page-request');
    const cacheContainer = document.getElementById('cache-memory');
    const cacheHitsSpan = document.getElementById('cache-hits');
    const cacheMissesSpan = document.getElementById('cache-misses');
    const hitRateSpan = document.getElementById('hit-rate');
    const logContainer = document.getElementById('simulation-log');

    // --- Cache State ---
    const CACHE_SIZE = 4;
    let cache = [];
    let cacheHistory = {};
    let hits = 0;
    let misses = 0;
    let isProcessing = false;

    /**
     * Handles the user's request to access a page.
     */
    async function handleRequest() {
        if (isProcessing) return;
        const pageId = pageInput.value.trim().toUpperCase();
        if (!pageId) return;

        isProcessing = true;
        clearHighlights();

        if (cache.includes(pageId)) {
            // --- CACHE HIT ---
            hits++;
            logMessage(`HIT: Page "${pageId}" found in cache.`, 'hit');
            updateHistory(pageId, true);
            highlightBlock(pageId, 'hit');
        } else {
            // --- CACHE MISS ---
            misses++;
            logMessage(`MISS: Page "${pageId}" not in cache.`, 'miss');
            let animationDelay = 0;

            if (cache.length >= CACHE_SIZE) {
                // Cache is full, need to evict
                const pageToEvict = await predictEviction();
                logMessage(`EVICT: Policy selected page "${pageToEvict}" for eviction.`, 'evict');

                highlightEvicted(pageToEvict);

                const evictIndex = cache.indexOf(pageToEvict);

                // Validation logic
                if (evictIndex >= 0) {
                    cache[evictIndex] = pageId;
                } else {
                    // Failsafe: The policy returned a page not in the cache.
                    const fallbackPage = cache[0];
                    logMessage(`WARN: Policy returned invalid page "${pageToEvict}". Evicting "${fallbackPage}" instead.`, 'evict');
                    highlightEvicted(fallbackPage);
                    cache[0] = pageId;
                }
                animationDelay = 300; // Wait for eviction animation
            } else {
                // Cache is not full, just add
                cache.push(pageId);
                logMessage(`ADD: Adding page "${pageId}" to cache.`, 'info');
            }

            updateHistory(pageId, false);

            setTimeout(() => {
                highlightBlock(pageId, 'miss');
            }, animationDelay);
        }

        // Update UI
        updateStatistics();
        pageInput.value = ''; // Clear input
        pageInput.focus();

        setTimeout(() => {
            isProcessing = false;
        }, 350);
    }

    /**
     * Predicts which page to evict based on the selected policy.
     */
    async function predictEviction() {
        // Get the selected policy from the dropdown
        const policy = document.getElementById('policy-select')?.value || 'LRU';

        if (policy === 'LRU') {
            return evictLRU();
        }
        if (policy === 'MRU') {
            return evictMRU();
        }

        if (policy === 'ML_SERVER') {
            logMessage("INFO: Requesting eviction from server...", 'info');
            try {
                // Prepare feature data for all cached pages
                const features = [];
                const pageIds = [];
                const now = Date.now();

                // compute recency ranks (1 = most recent)
                const pagesWithLast = cache.map(p => {
                    const s = cacheHistory[p] || { frequency: 0, lastUsed: 0 };
                    return { page: p, lastUsed: s.lastUsed || 0 };
                });
                // sort descending by lastUsed (recent first)
                pagesWithLast.sort((a, b) => b.lastUsed - a.lastUsed);
                const recencyRankMap = {};
                pagesWithLast.forEach((obj, idx) => {
                    recencyRankMap[obj.page] = idx + 1; 
                });

                for (const pageId of cache) {
                    const stats = cacheHistory[pageId] || { frequency: 0, lastUsed: 0 };
                    const last_access_time = now - (stats.lastUsed || now);
                    const access_count = stats.frequency || 0;
                    const recency_rank = recencyRankMap[pageId] || cache.length;
                    const access_type = (stats.access_type !== undefined) ? stats.access_type : 0; 
                    const cache_item = (() => {
                        const n = parseInt(pageId, 10);
                        if (!isNaN(n)) return n;
                        let h = 0;
                        for (let i = 0; i < pageId.length; i++) {
                            h = (h * 31 + pageId.charCodeAt(i)) >>> 0;
                        }
                        return h % 1000;
                    })();

                    features.push({
                        last_access_time,
                        access_count,
                        recency_rank,
                        access_type,
                        cache_item
                    });
                    pageIds.push(pageId);
                }

                // Send to Flask ML server
                const resp = await fetch('/predict-evict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ pages: features, pageIds })
                });

                if (resp.ok) {
                    const data = await resp.json();
                    if (data && data.evict) {
                        logMessage(`SERVER: ML model chose "${data.evict}".`, 'info');
                        return data.evict;
                    } else {
                        throw new Error('Server returned no eviction item');
                    }
                } else {
                    const err = await resp.json().catch(() => ({}));
                    throw new Error(err.error || 'Server responded poorly');
                }
            } catch (err) {
                console.warn('Server ML call failed, falling back to LFU', err);
                logMessage(`ERROR: Server call failed. Falling back to LFU.`, 'evict');
                return evictLFU(); // fallback method
            }
        }

        // Default fallback
        return evictLRU();
    }

    /**
     * Evicts Least Recently Used page.
     */
    function evictLRU() {
        let pageToEvict = null;
        let oldest = Infinity;
        for (const p of cache) {
            const s = cacheHistory[p] || { lastUsed: 0 };
            if (s.lastUsed < oldest) {
                oldest = s.lastUsed;
                pageToEvict = p;
            }
        }
        return pageToEvict || cache[0]; // Failsafe
    }

    /**
     * Evicts Most Recently Used page.
     */
    function evictMRU() {
        let pageToEvict = null;
        let newest = -Infinity;
        for (const p of cache) {
            const s = cacheHistory[p] || { lastUsed: 0 };
            if (s.lastUsed > newest) {
                newest = s.lastUsed;
                pageToEvict = p;
            }
        }
        return pageToEvict || cache[0]; // Failsafe
    }

    /**
     * Evicts Least Frequently Used page.
     * This is used as a fallback if the server call fails.
     */
    function evictLFU() {
        if (!cache || cache.length === 0) return null;

        let pageToEvict = null;
        let lowestFreq = Infinity;
        let oldestTime = Infinity;

        for (const pageId of cache) {
            const stats = cacheHistory[pageId] || { frequency: 0, lastUsed: 0 };

            if (stats.frequency < lowestFreq) {
                lowestFreq = stats.frequency;
                oldestTime = stats.lastUsed || 0;
                pageToEvict = pageId;
            } else if (stats.frequency === lowestFreq) {
                if ((stats.lastUsed || 0) < oldestTime) {
                    oldestTime = stats.lastUsed || 0;
                    pageToEvict = pageId;
                }
            }
        }
        logMessage(`LFU fallback evicted "${pageToEvict}"`, 'evict');
        return pageToEvict || cache[0]; // Failsafe
    }

    /**
     * Updates the history log for a page (our "model's" data).
     */
    function updateHistory(pageId, isHit) {
        if (!cacheHistory[pageId]) {
            cacheHistory[pageId] = { frequency: 0, lastUsed: 0 };
        }
        cacheHistory[pageId].frequency++;
        cacheHistory[pageId].lastUsed = Date.now();
    }

    // --- UI/Helper Functions (No changes below this line) ---

    function renderCache() {
        cacheContainer.innerHTML = '';
        for (let i = 0; i < CACHE_SIZE; i++) {
            const block = document.createElement('div');
            block.classList.add('cache-block');
            if (cache[i]) {
                block.textContent = cache[i];
                block.dataset.pageId = cache[i];
            } else {
                block.textContent = 'Empty';
                block.classList.add('empty');
            }
            cacheContainer.appendChild(block);
        }
    }

    function updateStatistics() {
        cacheHitsSpan.textContent = hits;
        cacheMissesSpan.textContent = misses;
        const total = hits + misses;
        const rate = (total === 0) ? 0 : (hits / total) * 100;
        hitRateSpan.textContent = `${rate.toFixed(1)}%`;
    }

    function logMessage(message, type = 'info') {
        if (logContainer.querySelector(".log-info")) {
            logContainer.innerHTML = '';
        }
        const p = document.createElement('p');
        const time = new Date().toLocaleTimeString().split(' ')[0];
        p.textContent = `[${time}] ${message}`;
        p.classList.add(`log-${type}`);
        logContainer.prepend(p);
    }

    function highlightBlock(pageId, type) {
        renderCache();
        const block = cacheContainer.querySelector(`[data-page-id="${pageId}"]`);
        if (block) {
            block.classList.add(type === 'hit' ? 'highlight-hit' : 'highlight-miss');
        }
    }

    function highlightEvicted(pageId) {
        const block = cacheContainer.querySelector(`[data-page-id="${pageId}"]`);
        if (block) {
            block.classList.add('highlight-evict');
        }
    }

    function clearHighlights() {
        document.querySelectorAll('.cache-block').forEach(block => {
            block.classList.remove('highlight-hit', 'highlight-miss', 'highlight-evict');
        });
    }

    // --- Event Listeners ---
    requestBtn.addEventListener('click', handleRequest);
    pageInput.addEventListener('keyup', (e) => {
        if (e.key === 'Enter') {
            handleRequest();
        }
    });

    // --- Initial Render ---
    renderCache();
});
