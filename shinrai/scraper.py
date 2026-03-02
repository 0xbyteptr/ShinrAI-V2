import os
import random
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from collections import defaultdict
import logging

import requests
from bs4 import BeautifulSoup


class WebScraper:
    """Advanced web scraper with intelligent content extraction and multithreading"""

    def __init__(self, 
                 max_depth: int = 3, 
                 respect_robots: bool = True,
                 max_workers: int = 5,
                 rate_limit: float = 1.0,
                 timeout: int = 10,
                 max_retries: int = 2,
                 proxies: Optional[Dict[str, str]] = None,
                 user_agents: Optional[List[str]] = None):
        """
        Initialize the web scraper.
        
        Args:
            max_depth: Maximum crawling depth
            respect_robots: Whether to respect robots.txt
            max_workers: Maximum number of concurrent threads
            rate_limit: Minimum time between requests to same domain (seconds)
            timeout: Request timeout in seconds
            max_retries: Number of times to retry when a request fails or returns
                a 403.  The UA will be rotated between attempts.
            proxies: Optional proxy configuration to pass to requests/coudscraper.
            user_agents: A list of user-agent strings to rotate on retries.
        """
        self.max_depth = max_depth
        self.respect_robots = respect_robots
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.max_retries = max_retries
        # allow proxies from constructor or environment variables
        self.proxies = proxies or {
            k.lower(): v for k, v in {
                'http': os.environ.get('HTTP_PROXY'),
                'https': os.environ.get('HTTPS_PROXY')
            }.items() if v
        }
        # default UA list if none provided
        self.user_agents = user_agents or [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/117.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15',
            'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:115.0) '
            'Gecko/20100101 Firefox/115.0'
        ]
        
        # Thread-safe collections
        self.visited_urls: Set[str] = set()
        self.visited_lock = threading.Lock()
        
        self.scraped_data: List[Dict[str, Any]] = []
        self.data_lock = threading.Lock()
        
        # Rate limiting per domain
        self.last_request_time: Dict[str, float] = defaultdict(float)
        self.rate_lock = threading.Lock()
        
        # Queue for URLs to process
        self.url_queue = queue.PriorityQueue()  # Priority queue for BFS-like crawling
        
        # Session per thread (thread-local)
        self.thread_local = threading.local()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'pages_scraped': 0,
            'errors': 0,
            'total_bytes': 0,
            'start_time': None,
            'end_time': None
        }
        self.stats_lock = threading.Lock()

    def get_session(self) -> requests.Session:
        """Get or create a thread-local session.

        If the `cloudscraper` package is available we use it to get a session that
        can bypass Cloudflare/anti-bot protections. Otherwise fall back to plain
        :class:`requests.Session` with a basic browser user agent.  This makes the
        scraper more resilient to 403 errors when sites block simple bots.
        """
        if not hasattr(self.thread_local, "session"):
            try:
                import cloudscraper
                sess = cloudscraper.create_scraper()
                # cloudscraper already sets a User-Agent, but we'll override it to a
                # recent Chrome UA to maximise compatibility.
                sess.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/117.0.0.0 Safari/537.36'
                })
                self.logger.info("Using cloudscraper session to bypass anti-bot protections")
                self.thread_local.session = sess
            except ImportError:
                # fallback to normal requests
                self.thread_local.session = requests.Session()
                self.thread_local.session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/91.0.4472.124 Safari/537.36'
                })
            # if any proxies were specified for this scraper, apply them to the
            # session. requests and cloudscraper both understand the same dict
            # format.
            if self.proxies:
                self.thread_local.session.proxies.update(self.proxies)
        return self.thread_local.session

    def can_fetch(self, url: str) -> bool:
        """Check if we can fetch the URL based on rate limiting."""
        if not self.rate_limit:
            return True
            
        domain = urlparse(url).netloc
        with self.rate_lock:
            last_time = self.last_request_time.get(domain, 0)
            time_since_last = time.time() - last_time
            if time_since_last < self.rate_limit:
                return False
            self.last_request_time[domain] = time.time()
        return True

    def is_visited(self, url: str) -> bool:
        """Check if URL has been visited (thread-safe)."""
        with self.visited_lock:
            return url in self.visited_urls

    def mark_visited(self, url: str):
        """Mark URL as visited (thread-safe)."""
        with self.visited_lock:
            self.visited_urls.add(url)

    def add_scraped_data(self, data: Dict[str, Any]):
        """Add scraped data to results (thread-safe)."""
        with self.data_lock:
            self.scraped_data.append(data)
            self.stats['pages_scraped'] += 1

    def update_stats(self, **kwargs):
        """Update statistics (thread-safe)."""
        with self.stats_lock:
            for key, value in kwargs.items():
                if key in self.stats:
                    self.stats[key] += value

    def _make_request(self, url: str) -> requests.Response:
        """Perform HTTP GET with retry, UA rotation and optional proxies.

        Raises the final exception if all retries fail.
        """
        session = self.get_session()
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                resp = session.get(url, timeout=self.timeout, proxies=self.proxies or None)
                if resp.status_code == 403 and attempt < self.max_retries - 1:
                    self.logger.warning(
                        f"403 for {url}, rotating User-Agent and retrying (attempt {attempt+1})"
                    )
                    session.headers.update({
                        'User-Agent': random.choice(self.user_agents)
                    })
                    time.sleep(self.rate_limit or 1)
                    continue
                return resp
            except Exception as e:  # includes timeout/conn errors
                last_exc = e
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"Error fetching {url} ({e}), retrying")
                    time.sleep(self.rate_limit or 1)
                    continue
                raise
        # if we got here something went wrong
        raise last_exc  # type: ignore

    def scrape_page_worker(self, url: str, depth: int, domain: str) -> Optional[Dict]:
        """Worker function to scrape a single page."""
        try:
            # Check rate limiting; if we're too soon, wait rather than requeue.
            while not self.can_fetch(url):
                time.sleep(self.rate_limit or 0.1)

            # Perform request with retry logic
            response = self._make_request(url)
            
            with self.stats_lock:
                self.stats['total_bytes'] += len(response.content)

            if response.status_code != 200:
                # If the site is actively blocking us, offer a hint in the log.
                if response.status_code == 403:
                    self.logger.error(
                        f"Access forbidden (403) for {url}. "
                        "You may need to supply proxies, rotate IPs, or fetch the "
                        "content manually and train from a local file."
                    )
                else:
                    self.logger.error(f"Non-200 status code {response.status_code} for {url}")
                self.update_stats(errors=1)
                return None

            # Parse page
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
                element.decompose()

            # Extract metadata, convert any bs4 strings to plain str
            title = ''
            if soup.title and soup.title.string is not None:
                title = str(soup.title.string)

            meta_description = ''
            meta_tag = soup.find('meta', attrs={'name': 'description'})
            if meta_tag:
                meta_description = str(meta_tag.get('content', ''))

            # Extract main content
            main_content = self._extract_main_content(soup)

            # Extract links for further crawling; we no longer restrict to the
            # original domain because many useful scrapes (e.g. search engine
            # results) link out to other hosts.  We still avoid common media
            # file types to save time.
            new_links = []
            if depth < self.max_depth:
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(url, href)

                    # ignore anchors and javascript pseudo-links
                    if full_url.startswith('javascript:') or full_url.startswith('#'):
                        continue

                    # ignore common binary/media extensions
                    if any(ext in full_url.lower() for ext in
                           ['.jpg', '.png', '.zip', '.mp3', '.mp4', '.css', '.js']):
                        continue

                    if not self.is_visited(full_url):
                        new_links.append(full_url)

            # Extract headings for structure (always convert to str)
            headings = []
            for tag in ['h1', 'h2', 'h3']:
                for heading in soup.find_all(tag):
                    headings.append(str(heading.get_text().strip()))

            page_data = {
                'url': url,
                'title': title,
                'meta_description': meta_description,
                'content': main_content,
                'headings': headings,
                'links': new_links[:20],  # Limit links per page
                'depth': depth,
                'timestamp': datetime.now().isoformat(),
                'content_length': len(main_content)
            }

            self.add_scraped_data(page_data)
            self.logger.info(f"Scraped {url} - {len(main_content)} chars at depth {depth}")

            return page_data

        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout scraping {url}")
            self.update_stats(errors=1)
            return None
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Connection error scraping {url}")
            self.update_stats(errors=1)
            return None
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {e}")
            self.update_stats(errors=1)
            return None

    def scrape(self, start_url: str, max_pages: int = 100) -> List[Dict[str, Any]]:
        """
        Scrape website (and linked pages) with intelligent content extraction
        using multithreading.

        The scraper no longer confines itself to the start URL's domain; it
        will follow any HTTP/HTTPS link it discovers (except for common media
        file types).  This makes it easier to crawl search results or sites
        that redirect between subdomains.

        Args:
            start_url: URL to start scraping from
            max_pages: Maximum number of pages to scrape

        Returns:
            List of dictionaries containing scraped page data
        """
        self.logger.info(f"Starting multithreaded scrape from {start_url}")
        self.logger.info(f"Max pages: {max_pages}, Workers: {self.max_workers}")
        
        self.stats['start_time'] = time.time()
        
        # Reset state for new scrape
        self.visited_urls.clear()
        self.scraped_data.clear()
        self.url_queue = queue.PriorityQueue()
        
        domain = urlparse(start_url).netloc
        
        # Add initial URL
        self.url_queue.put((0, start_url))
        
        # Create thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            # Submit initial batch of URLs
            for _ in range(min(self.max_workers, max_pages)):
                try:
                    depth, url = self.url_queue.get(timeout=1)
                    if not self.is_visited(url):
                        self.mark_visited(url)
                        future = executor.submit(self.scrape_page_worker, url, depth, domain)
                        futures.append(future)
                    self.url_queue.task_done()
                except queue.Empty:
                    break
            
            # Process results and keep refilling from queue
            completed = 0
            while completed < max_pages and (futures or not self.url_queue.empty()):
                # If we have capacity and there are queued URLs, submit them
                while (len(futures) < self.max_workers and
                       not self.url_queue.empty() and
                       completed < max_pages):
                    try:
                        depth, url = self.url_queue.get_nowait()
                    except queue.Empty:
                        break
                    if not self.is_visited(url):
                        self.mark_visited(url)
                        futures.append(
                            executor.submit(self.scrape_page_worker, url, depth, domain)
                        )
                    self.url_queue.task_done()
                
                if not futures:
                    # nothing currently running; loop will check queue again
                    time.sleep(0.1)
                    continue

                # Wait for any future to complete
                done, futures = self._wait_for_futures(futures)
                
                for future in done:
                    completed += 1
                    try:
                        page_data = future.result(timeout=5)
                        
                        if page_data and page_data['depth'] < self.max_depth:
                            # enqueue new URLs from this page instead of submitting directly
                            for link in page_data.get('links', []):
                                if not self.is_visited(link) and completed < max_pages:
                                    self.url_queue.put((page_data['depth'] + 1, link))
                    except Exception as e:
                        self.logger.error(f"Error processing future result: {e}")
                        self.update_stats(errors=1)
                
                # Log progress
                if completed % 10 == 0:
                    self.logger.info(f"Progress: {completed}/{max_pages} pages scraped")
        
        self.stats['end_time'] = time.time()
        duration = self.stats['end_time'] - self.stats['start_time']
        
        # Log statistics
        self.logger.info("=" * 50)
        self.logger.info("SCRAPING COMPLETED - STATISTICS")
        self.logger.info("=" * 50)
        self.logger.info(f"Pages scraped: {self.stats['pages_scraped']}")
        self.logger.info(f"Errors: {self.stats['errors']}")
        self.logger.info(f"Total data: {self.stats['total_bytes'] / 1024:.2f} KB")
        self.logger.info(f"Time taken: {duration:.2f} seconds")
        self.logger.info(f"Average speed: {self.stats['pages_scraped'] / duration:.2f} pages/sec")
        self.logger.info("=" * 50)
        
        return self.scraped_data

    def _wait_for_futures(self, futures: List) -> Tuple[List, List]:
        """Wait for any future to complete and return completed and remaining futures."""
        from concurrent.futures import wait, FIRST_COMPLETED
        
        done, remaining = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
        return list(done), list(remaining)

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from soup."""
        # Try common content containers
        content_candidates = []

        # Look for main content areas
        for selector in ['main', 'article', '#content', '.content', '.post', '.article']:
            elements = soup.select(selector)
            content_candidates.extend(elements)

        # If no specific content found, use body
        if not content_candidates:
            content_candidates = [soup.body] if soup.body else [soup]

        # Extract text from best candidate
        text = ' '.join([c.get_text() for c in content_candidates])

        # Clean text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text

    def scrape_parallel(self, urls: List[str], max_workers: int = None) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs in parallel.
        
        Args:
            urls: List of URLs to scrape
            max_workers: Maximum number of worker threads (defaults to self.max_workers)
            
        Returns:
            List of scraped page data
        """
        workers = max_workers or self.max_workers
        self.logger.info(f"Parallel scraping {len(urls)} URLs with {workers} workers")
        
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all scraping tasks
            future_to_url = {
                executor.submit(self._scrape_single_url, url): url 
                for url in urls
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result(timeout=self.timeout + 5)
                    if result:
                        results.append(result)
                        self.logger.info(f"Successfully scraped {url}")
                except Exception as e:
                    self.logger.error(f"Error scraping {url}: {e}")
                    self.update_stats(errors=1)
        
        return results

    def _scrape_single_url(self, url: str) -> Optional[Dict]:
        """Scrape a single URL (for parallel scraping)."""
        try:
            session = self.get_session()
            response = session.get(url, timeout=self.timeout)
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Extract content
            title = soup.title.string if soup.title else ''
            content = self._extract_main_content(soup)
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'timestamp': datetime.now().isoformat(),
                'content_length': len(content)
            }
            
        except Exception as e:
            self.logger.error(f"Error in _scrape_single_url for {url}: {e}")
            return None