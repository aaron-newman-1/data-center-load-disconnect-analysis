"""
PJM DataMiner2 — July 10, 2024 Load Disconnection/Reconnection Event
=====================================================================
Pulls real-time load and transmission constraint data from the PJM
DataMiner2 API and produces an annotated visualization of the
disconnection/reconnection event.

Usage:
    pip install requests pandas matplotlib python-dotenv

    Set your API key via environment variable or pass it directly:
        export PJM_API_KEY="your_key_here"
        python pjm_load_event_analysis.py

    Or edit the API_KEY constant below.
"""