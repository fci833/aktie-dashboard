"""Parallel data fetching for watchlist & screener"""
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable, Any


def parallel_fetch(tickers: List[str], fetch_fn: Callable, max_workers: int = 8,
                   show_progress: bool = True) -> Dict[str, Any]:
    """
    Fetcher data parallelt for liste af tickers.
    
    Args:
        tickers: liste af tickers
        fetch_fn: funktion der tager én ticker og returnerer data
        max_workers: antal parallelle threads (default 8)
        show_progress: vis progress bar
    
    Returns:
        dict {ticker: data}
    """
    results = {}
    if not tickers:
        return results

    progress_bar = None
    status = None
    if show_progress:
        progress_bar = st.progress(0.0)
        status = st.empty()

    completed = 0
    total = len(tickers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(fetch_fn, t): t for t in tickers
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result(timeout=30)
            except Exception as e:
                results[ticker] = {"error": str(e)}

            completed += 1
            if progress_bar:
                progress_bar.progress(completed / total)
            if status:
                status.text(f"⚡ Henter data... {completed}/{total} ({ticker})")

    if progress_bar:
        progress_bar.empty()
    if status:
        status.empty()

    return results


def parallel_fetch_silent(tickers: List[str], fetch_fn: Callable,
                          max_workers: int = 8) -> Dict[str, Any]:
    """Silent version uden progress bar"""
    return parallel_fetch(tickers, fetch_fn, max_workers, show_progress=False)
