import os
import json
import time
import ccxt
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from dotenv import load_dotenv


#region live performance: helper

def floor_to_hour(ts_ms):
    hour_ms = 3600000
    return (ts_ms // hour_ms) * hour_ms


def snap_to_hour(ts_ms):
    # Round to nearest hour in milliseconds
    hour_ms = 3600000
    return int(round(ts_ms / hour_ms) * hour_ms)


def redact_sensitive(obj):
    if isinstance(obj, dict):
        return {k: redact_sensitive(v) for k, v in obj.items() if k not in ('accountId', 'userId')}
    if isinstance(obj, list):
        return [redact_sensitive(v) for v in obj]
    return obj


def append_json(filename, records, proof_dir):
    if not records:
        return
    file_path = os.path.join(proof_dir, filename)
    # Read existing records
    existing_records = _read_all_json(file_path)
    # Append new records
    existing_records.extend(records)
    # Write back all records as JSON array
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump([redact_sensitive(rec) for rec in existing_records], f, ensure_ascii=False, indent=2)


def convert_jsonl_to_json(proof_dir):
    """One-time conversion function to migrate existing JSONL files to JSON arrays."""
    import shutil

    jsonl_files = ['orders.jsonl', 'closed_pnl.jsonl', 'funding_fees.jsonl']
    json_files = ['orders.json', 'closed_pnl.json', 'funding_fees.json']

    for jsonl_file, json_file in zip(jsonl_files, json_files):
        jsonl_path = os.path.join(proof_dir, jsonl_file)
        json_path = os.path.join(proof_dir, json_file)

        if os.path.exists(jsonl_path):
            print(f"Converting {jsonl_file} to {json_file}...")
            # Read existing JSONL data
            records = _read_all_jsonl(jsonl_path)
            # Write as JSON array
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            # Backup and remove old file
            backup_path = jsonl_path + '.backup'
            shutil.move(jsonl_path, backup_path)
            print(f"Converted {len(records)} records. Original backed up to {backup_path}")
        else:
            print(f"No {jsonl_file} found to convert")


def _read_all_jsonl(path):
    """Legacy function for reading JSONL files during conversion."""
    out = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return out


def _read_all_json(path):
    """Read JSON array file and return list of records."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def create_exchange(api_key, api_secret):
    temp_exchange = ccxt.bybit()
    server_time = temp_exchange.fetch_time()
    local_time = int(time.time() * 1000)
    time_diff = server_time - local_time
    print(f"Time difference: {time_diff}ms")

    exchange = ccxt.bybit({
        'apiKey': api_key,
        'secret': api_secret,
        'options': {
            'defaultType': 'linear',
            'timeDifference': abs(time_diff),
            'recvWindow': 10000,
        }
    })
    exchange.verbose = False
    return exchange


def fetch_closed_pnl(exchange, start_time):
    all_results = []
    current_start_time = start_time
    current_time = int(datetime.now().timestamp()) * 1000

    print("Fetching closed PnL data...")

    while current_start_time < current_time:
        current_end_time = min(current_start_time + (7 * 24 * 60 * 60 * 1000), current_time)

        params = {
            'accountType': 'UNIFIED',
            'category': 'linear',
            'settleCoin': 'USDT',
            'startTime': current_start_time,
            'endTime': current_end_time,
            'limit': 100
        }

        # Page through results within the 7-day window using nextPageCursor
        next_cursor = None
        total_added_in_window = 0
        while True:
            req = dict(params)
            if next_cursor:
                req['cursor'] = next_cursor
            response = exchange.private_get_v5_position_closed_pnl(req)
            result_obj = response.get('result', {}) if isinstance(response, dict) else {}
            result_list = result_obj.get('list', []) or []
            if not result_list:
                break
            # Remove duplicates conservatively by (orderId, updatedTime)
            for record in result_list:
                try:
                    if not any(
                        existing.get('updatedTime') == record.get('updatedTime') and
                        existing.get('orderId') == record.get('orderId')
                        for existing in all_results
                    ):
                        all_results.append(record)
                        total_added_in_window += 1
                except Exception:
                    continue
            next_cursor = result_obj.get('nextPageCursor') or None
            if not next_cursor:
                break

        print(f"Added {total_added_in_window} records")
        current_start_time = current_end_time
        time.sleep(0.1)

    print(f"Total records: {len(all_results)}")
    return all_results


def fetch_funding_fees(exchange, start_time):
    """Fetch funding fee timeline"""
    funding_timeline = []
    current_start_time = start_time
    current_time = int(datetime.now().timestamp()) * 1000

    print("Fetching funding fees...")

    while current_start_time < current_time:
        current_end_time = min(current_start_time + (7 * 24 * 60 * 60 * 1000), current_time)

        params = {
            'accountType': 'UNIFIED',
            'category': 'linear',
            'type': 'SETTLEMENT',
            'startTime': current_start_time,
            'endTime': current_end_time,
            'limit': 50
        }

        # Page through account transaction logs (funding) within the window
        next_cursor = None
        total_added_in_window = 0
        page_idx = 0
        while True:
            req = dict(params)
            if next_cursor:
                req['cursor'] = next_cursor
            response = exchange.private_get_v5_account_transaction_log(req)
            result_obj = response.get('result', {}) if isinstance(response, dict) else {}
            result_list = result_obj.get('list', []) or []
            if not result_list:
                break
            page_idx += 1
            added_this_page = 0
            for record in result_list:
                try:
                    # Prefer explicit funding field; some entries may carry bizType
                    if record.get('funding') and float(record['funding']) != 0:
                        funding_timeline.append({
                            'timestamp': int(record['transactionTime']),
                            'symbol': record.get('symbol'),
                            'funding': float(record['funding'])
                        })
                        total_added_in_window += 1
                        added_this_page += 1
                except Exception:
                    continue
            # print(f"Funding: window [{current_start_time}..{current_end_time}) page {page_idx} added {added_this_page}")
            next_cursor = result_obj.get('nextPageCursor') or None
            if not next_cursor:
                break

        print(f"Funding: window total added {total_added_in_window}")
        current_start_time = current_end_time
        time.sleep(0.1)

    # Sort and remove duplicates
    funding_timeline.sort(key=lambda x: x['timestamp'])
    unique_funding = list({(f['timestamp'], f['symbol'], f['funding']): f for f in funding_timeline}.values())

    print(f"Funding records: {len(unique_funding)}")
    return unique_funding


def fetch_order_history(exchange, start_time):
    """Fetch order history (filled orders) with pagination over 7-day windows."""
    all_orders = []
    current_start_time = start_time
    current_time = int(datetime.now().timestamp()) * 1000
    print("Fetching order history...")
    while current_start_time < current_time:
        current_end_time = min(current_start_time + (7 * 24 * 60 * 60 * 1000), current_time)
        params = {
            'category': 'linear',
            'startTime': current_start_time,
            'endTime': current_end_time,
            'limit': 50
        }
        next_cursor = None
        total_added_in_window = 0
        page_idx = 0
        while True:
            req = dict(params)
            if next_cursor:
                req['cursor'] = next_cursor
            resp = exchange.private_get_v5_order_history(req)
            result_obj = resp.get('result', {}) if isinstance(resp, dict) else {}
            lst = result_obj.get('list', []) or []
            if not lst:
                break
            page_idx += 1
            added_this_page = 0
            for r in lst:
                try:
                    if r.get('orderStatus') != 'Filled':
                        continue
                    all_orders.append(r)
                    total_added_in_window += 1
                    added_this_page += 1
                except Exception:
                    continue
            # print(f"Orders: window [{current_start_time}..{current_end_time}) page {page_idx} added {added_this_page}")
            next_cursor = result_obj.get('nextPageCursor') or None
            if not next_cursor:
                break
        print(f"Orders: window total added {total_added_in_window}")
        current_start_time = current_end_time
        time.sleep(0.1)
    print(f"Total orders: {len(all_orders)}")
    return all_orders


def _read_max_timestamp_from_json(file_path, candidate_keys):
    """Scan a JSON array file and return the maximum integer timestamp found across candidate keys.
    candidate_keys may be a list like ['updatedTime', 'createdTime'].
    Returns None if file does not exist or no valid timestamps found.
    """
    try:
        if not os.path.exists(file_path):
            return None
        records = _read_all_json(file_path)
        max_ts = None
        for obj in records:
            for key in candidate_keys:
                val = obj.get(key)
                if val is None:
                    continue
                try:
                    ts = int(val)
                    if (max_ts is None) or (ts > max_ts):
                        max_ts = ts
                except Exception:
                    continue
        return max_ts
    except Exception:
        return None


def update_live_trading_data(exchange, start_time_ms, proof_dir, load_only):
    """Fetch only new data since the last saved JSON entries, append to disk, and return newly fetched records.

    When load_only=True, skip fetching and just load existing data from proof_dir.

    Returns: (orders_new, closed_pnl_new, funding_fees_new)
    """
    os.makedirs(proof_dir, exist_ok=True)

    if not load_only:
        # Determine per-dataset effective start time by reading last saved timestamps
        orders_file = os.path.join(proof_dir, 'orders.json')
        closed_pnl_file = os.path.join(proof_dir, 'closed_pnl.json')
        funding_fees_file = os.path.join(proof_dir, 'funding_fees.json')

        last_orders_ts = _read_max_timestamp_from_json(orders_file, ['updatedTime', 'createdTime'])
        last_closed_ts = _read_max_timestamp_from_json(closed_pnl_file, ['updatedTime'])
        last_funding_ts = _read_max_timestamp_from_json(funding_fees_file, ['timestamp'])

        eff_orders_start = max(start_time_ms, (last_orders_ts + 1) if last_orders_ts is not None else start_time_ms)
        eff_closed_start = max(start_time_ms, (last_closed_ts + 1) if last_closed_ts is not None else start_time_ms)
        eff_funding_start = max(start_time_ms, (last_funding_ts + 1) if last_funding_ts is not None else start_time_ms)

        # Fetch new data from exchange
        closed_pnl_new = fetch_closed_pnl(exchange, eff_closed_start)
        funding_fees_new = fetch_funding_fees(exchange, eff_funding_start)
        orders_new = fetch_order_history(exchange, eff_orders_start)

        # Append to disk using existing helpers
        append_json('orders.json', orders_new, proof_dir)
        append_json('closed_pnl.json', closed_pnl_new, proof_dir)
        append_json('funding_fees.json', funding_fees_new, proof_dir)

    # Read from disk
    closed_pnl = _read_all_json(os.path.join(proof_dir, 'closed_pnl.json'))
    funding_fees = _read_all_json(os.path.join(proof_dir, 'funding_fees.json'))
    orders = _read_all_json(os.path.join(proof_dir, 'orders.json'))
    closed_pnl.sort(key=lambda x: int(x['updatedTime']))

    return orders, closed_pnl, funding_fees


def accumulate_hourly_trading_fees(orders_list, hours_arr):
    per_hour = {int(h): 0.0 for h in hours_arr}
    for o in orders_list:
        try:
            ts = floor_to_hour(int(o.get('updatedTime') or o.get('createdTime')))
            # cumExecFee may be str; cumFeeDetail also available but sum is in cumExecFee
            fee = float(o.get('cumExecFee') or 0.0)
            if ts in per_hour:
                per_hour[ts] += fee
        except Exception:
            continue
    out = []
    running = 0.0
    for h in hours_arr:
        running += per_hour[int(h)]
        out.append(running)
    return np.array(out, dtype=np.float64)


def load_open_prices_from_disk(symbols, start_ts_ms, end_ts_ms, source="bybit"):
    if source == "bybit":
        base_dir = "../../Bybit/data/klines/1 hour"
    else:
        base_dir = "../Data/1 hour data new/trading klines"
    opens = {s: {} for s in symbols}
    for s in symbols:
        try:
            npz_path = os.path.join(base_dir, f"{s}.npz")
            with np.load(npz_path) as z:
                ts = z['timestamps']
                kl = z['klines']
            for t, row in zip(ts, kl):
                t = int(t)
                if t < start_ts_ms or t > end_ts_ms:
                    continue
                opens[s][snap_to_hour(t)] = float(row[0])
        except Exception as e:
            print(f"Open load skipped for {s}: {e}")
            continue
    return opens


def get_latest_kline_ts(symbols, source="bybit"):
    """Return the maximum available kline timestamp (ms) across provided symbols from disk."""
    if source == "bybit":
        base_dir = "../../Bybit/data/klines/1 hour"
    else:
        base_dir = "../Data/1 hour data new/trading klines"
    last_ts = []
    for s in symbols:
        try:
            npz_path = os.path.join(base_dir, f"{s}.npz")
            with np.load(npz_path) as z:
                ts = z['timestamps']
                if ts.size:
                    last_ts.append(int(ts[-1]))
        except Exception:
            continue
    return max(last_ts) if last_ts else None


def accumulate_hourly_unrealized_from_orders(orders_list, hours_arr, opens_by_symbol):
    state = {}
    orders_by_hour = {}
    for o in orders_list:
        try:
            ts = floor_to_hour(int(o.get('updatedTime') or o.get('createdTime')))
            orders_by_hour.setdefault(ts, []).append(o)
        except Exception:
            continue

    unreal = []
    for h in hours_arr:
        for o in orders_by_hour.get(int(h), []) or []:
            try:
                sym = o.get('symbol')
                side = o.get('side')
                reduce_only = bool(o.get('reduceOnly'))
                qty = float(o.get('cumExecQty') or 0.0)
                price = float(o.get('avgPrice') or 0.0)
                if not sym or qty == 0.0 or price == 0.0:
                    continue
                size_prev, vwap_prev = state.get(sym, (0.0, 0.0))
                sign = 1.0 if (side and side.upper() == 'BUY') else -1.0
                if not reduce_only:
                    size_new = size_prev + sign * qty
                    if size_prev == 0.0:
                        vwap_new = price
                    elif (size_prev > 0 and sign > 0) or (size_prev < 0 and sign < 0):
                        vwap_new = (abs(size_prev) * vwap_prev + qty * price) / (abs(size_prev) + qty)
                    else:
                        vwap_new = price
                        size_new = sign * max(0.0, abs(qty) - abs(size_prev))
                    state[sym] = (size_new, vwap_new)
                else:
                    state[sym] = (0.0, 0.0)
            except Exception:
                continue
        total = 0.0
        for sym, (sz, vp) in state.items():
            if sz == 0.0:
                continue
            price = opens_by_symbol.get(sym, {}).get(int(h))
            if price is None:
                continue
            total += sz * (price - vp)
        unreal.append(total)
    return np.array(unreal, dtype=np.float64)


def accumulate_hourly_closed_pnl(closed_list, hours_arr, start_balance):
    per_hour = {int(h): 0.0 for h in hours_arr}
    for e in closed_list:
        ts = floor_to_hour(int(e['updatedTime']))
        pnl = float(e['closedPnl'])
        if ts in per_hour:
            per_hour[ts] += pnl
    # cumulative
    out = []
    running = start_balance
    for h in hours_arr:
        running += per_hour[int(h)]
        out.append(running)
    return np.array(out, dtype=np.float64)


def accumulate_hourly_realized_from_orders(orders_list, hours_arr):
    """Reconstruct realized PnL from order history using avgPrice and cumExecQty.
    Assumes linear USDT perps; shorts have negative size.
    """
    # Per-symbol running state
    state = {}
    # Per-hour realized sums
    per_hour = {int(h): 0.0 for h in hours_arr}
    # Sort orders chronologically
    try:
        orders_sorted = sorted(orders_list, key=lambda o: int(o.get('updatedTime') or o.get('createdTime') or 0))
    except Exception:
        orders_sorted = orders_list

    for o in orders_sorted:
        try:
            sym = o.get('symbol')
            side = o.get('side')
            reduce_only = bool(o.get('reduceOnly'))
            qty = float(o.get('cumExecQty') or 0.0)
            price = float(o.get('avgPrice') or 0.0)
            if not sym or qty == 0.0 or price == 0.0:
                continue
            ts_hour = floor_to_hour(int(o.get('updatedTime') or o.get('createdTime')))
            size_prev, vwap_prev = state.get(sym, (0.0, 0.0))
            sign = 1.0 if (side and side.upper() == 'BUY') else -1.0

            if not reduce_only:
                # Open / increase position on side
                size_new = size_prev + sign * qty
                if size_prev == 0.0:
                    vwap_new = price
                elif (size_prev > 0 and sign > 0) or (size_prev < 0 and sign < 0):
                    vwap_new = (abs(size_prev) * vwap_prev + qty * price) / (abs(size_prev) + qty)
                else:
                    # Opposite without reduceOnly (flip). Realize old then open new.
                    realized = size_prev * (price - vwap_prev)
                    if ts_hour in per_hour:
                        per_hour[ts_hour] += realized
                    # New side remainder (assume full flip as per user flow rarely used)
                    vwap_new = price
                    size_new = sign * max(0.0, abs(qty) - abs(size_prev))
                state[sym] = (size_new, vwap_new)
            else:
                # Close position; per user's flow, full close
                realized = size_prev * (price - vwap_prev)
                if ts_hour in per_hour:
                    per_hour[ts_hour] += realized
                state[sym] = (0.0, 0.0)
        except Exception:
            continue

    # Cumulative over hours
    out = []
    running = 0.0
    for h in hours_arr:
        running += per_hour[int(h)]
        out.append(running)
    return np.array(out, dtype=np.float64)


def accumulate_hourly_funding(funding_list, hours_arr):
    per_hour = {int(h): 0.0 for h in hours_arr}
    for f in funding_list:
        ts = snap_to_hour(int(f['timestamp']))
        amt = float(f['funding'])
        if ts in per_hour:
            per_hour[ts] += amt
    out = []
    running = 0.0
    for h in hours_arr:
        running += per_hour[int(h)]
        out.append(running)
    return np.array(out, dtype=np.float64)


def build_initial_positions_from_orders(orders_list, cutoff_ts_ms, only_shorts=True):
    """Reconstruct open positions as of cutoff timestamp by replaying fills forward.
    Returns a dict compatible with the simulator's warm-start input:
    { symbol: { 'entryPrice': vwap, 'qty': abs(size) } }
    - size is signed (short < 0), VWAP reflects remaining open side.
    - reduceOnly closes reduce toward zero; VWAP preserved until zero.
    """
    cutoff = floor_to_hour(int(cutoff_ts_ms))
    try:
        orders_sorted = sorted(orders_list, key=lambda o: int(o.get('updatedTime') or o.get('createdTime') or 0))
    except Exception:
        orders_sorted = orders_list

    state = {}  # symbol -> (size_signed, vwap)
    for o in orders_sorted:
        try:
            ts_raw = int(o.get('updatedTime') or o.get('createdTime') or 0)
            if ts_raw > cutoff:
                break
            sym = o.get('symbol')
            if not sym:
                continue
            side = o.get('side')
            reduce_only = bool(o.get('reduceOnly'))
            qty = float(o.get('cumExecQty') or 0.0)
            price = float(o.get('avgPrice') or 0.0)
            if qty == 0.0 or price == 0.0:
                continue

            size_prev, vwap_prev = state.get(sym, (0.0, 0.0))
            sign = 1.0 if (side and side.upper() == 'BUY') else -1.0

            if reduce_only:
                # Move position toward zero; preserve VWAP until fully closed
                size_new = size_prev + sign * qty
                # If it crosses zero, clamp to zero and reset VWAP
                if (size_prev < 0 and size_new > 0) or (size_prev > 0 and size_new < 0):
                    state[sym] = (0.0, 0.0)
                elif abs(size_new) <= 1e-15:
                    state[sym] = (0.0, 0.0)
                else:
                    state[sym] = (size_new, vwap_prev)
                continue

            # Non-reduce-only
            if size_prev == 0.0:
                state[sym] = (sign * qty, price)
                continue

            same_side = (size_prev > 0 and sign > 0) or (size_prev < 0 and sign < 0)
            if same_side:
                size_abs_prev = abs(size_prev)
                size_new_signed = size_prev + sign * qty
                vwap_new = (size_abs_prev * vwap_prev + qty * price) / (size_abs_prev + qty)
                state[sym] = (size_new_signed, vwap_new)
            else:
                # Opposite without reduceOnly: partial close or flip
                delta_signed = sign * qty
                size_new_signed = size_prev + delta_signed
                if size_prev * sign < 0 and abs(delta_signed) < abs(size_prev):
                    # Partial close: keep VWAP
                    if abs(size_new_signed) <= 1e-15:
                        state[sym] = (0.0, 0.0)
                    else:
                        state[sym] = (size_new_signed, vwap_prev)
                else:
                    # Flip: realize prior and open remainder on new side at current price
                    remainder_signed = sign * max(0.0, abs(qty) - abs(size_prev))
                    if abs(remainder_signed) <= 1e-15:
                        state[sym] = (0.0, 0.0)
                    else:
                        state[sym] = (remainder_signed, price)
        except Exception:
            continue

    out = {}
    for sym, (sz, vp) in state.items():
        if abs(sz) <= 1e-15:
            continue
        if only_shorts and sz >= 0:
            continue
        out[sym] = {'entryPrice': float(vp), 'qty': float(abs(sz))}
    return out

#endregion

#region retrieve_live_performance

def retrieve_live_performance(start_date, end_date=None, start_balance=0, phase="all", positions_date=None, proof_dir='../Evaluation/1 hour data new, d6/live evaluation/proof', load_only=False):
    if not load_only:
        load_dotenv()
        api_key = os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_SECRET')
        exchange = create_exchange(api_key, api_secret)
    else:
        exchange = None

    start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    start_hour_ts = floor_to_hour(int(start_date.timestamp()) * 1000)
    if end_date is None:
        end_ts_now = int(datetime.now().timestamp()) * 1000
    else:
        end_ts_now = int(datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S').timestamp()) * 1000

    # Update data incrementally and get newly fetched records (or just load if load_only=True)
    orders, closed_pnl, funding_fees = update_live_trading_data(exchange, start_hour_ts, proof_dir=proof_dir, load_only=load_only)

    # Create hourly grid
    symbols = sorted({order['symbol'] for order in orders})
    latest_ts = get_latest_kline_ts(symbols, source="bybit") - 3600000 # subtract 1 hour to match simulator
    end_hour_ts = min(floor_to_hour(end_ts_now), floor_to_hour(latest_ts))
    hours_ts = np.arange(start_hour_ts, end_hour_ts + 1, 3600000, dtype=np.int64)
    hours_dt = [datetime.utcfromtimestamp(t/1000) for t in hours_ts]

    # get positions at positions_date
    if positions_date is not None:
        cutoff_ms = floor_to_hour(int(datetime.strptime(positions_date, '%Y-%m-%d %H:%M:%S').timestamp()) * 1000)
        cutoff_ms = min(cutoff_ms, end_hour_ts)
    else:
        cutoff_ms = end_hour_ts
    positions = build_initial_positions_from_orders(orders, cutoff_ms, only_shorts=True)
    print(f"{len(positions)} Positions at {datetime.utcfromtimestamp(cutoff_ms/1000)}: {positions}")

    hourly_cum_closed = accumulate_hourly_closed_pnl(closed_pnl, hours_ts, start_balance)
    hourly_cum_realized = accumulate_hourly_realized_from_orders(orders, hours_ts)
    hourly_cum_funding = accumulate_hourly_funding(funding_fees, hours_ts)
    hourly_cum_fees = accumulate_hourly_trading_fees(orders, hours_ts)
    opens_map = load_open_prices_from_disk(symbols, start_hour_ts, end_hour_ts, source="bybit")
    hourly_unrealized = accumulate_hourly_unrealized_from_orders(orders, hours_ts, opens_map)

    if phase in ("funding", "1"):
        equity_hourly = start_balance + hourly_cum_funding
    elif phase in ("fees", "2"):
        equity_hourly = start_balance - hourly_cum_fees
    elif phase in ("realized", "3"):
        equity_hourly = start_balance + hourly_cum_realized
    elif phase in ("unrealized", "4"):
        equity_hourly = start_balance + hourly_unrealized
    else:
        equity_hourly = start_balance - hourly_cum_fees + hourly_cum_funding + hourly_cum_realized + hourly_unrealized
        
    # Summary
    print(f"\nTotal closes: {len(closed_pnl)}")
    print(f"Original Total PnL: {start_balance - hourly_cum_realized[-1]:.6f}")
    print(f"Total funding fees removed: {hourly_cum_funding[-1]:.6f}")

    # Closed PnL
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.step(hours_dt, hourly_cum_closed, where='post', color='green')
    ax1.set_title('Closed PnL')
    # Equity plot (composition depends on phase)
    ax2.step(hours_dt, equity_hourly, where='post', color='green')
    ax2.set_title('Equity (phase dependent)')

    # Fees and Funding
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))
    ax3.step(hours_dt, -hourly_cum_fees, where='post', color='red')
    ax3.set_title('Cumulative Fees')
    ax4.step(hours_dt, hourly_cum_funding, where='post', color='red')
    ax4.set_title('Cumulative Funding')

    # Realized and Unrealized PnL
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 4))
    ax5.step(hours_dt, hourly_cum_realized, where='post', color='blue')
    ax5.set_title('Cumulative Realized PnL')
    ax6.step(hours_dt, hourly_unrealized, where='post', color='blue')
    ax6.set_title('Cumulative Unrealized PnL')

    # Format dates
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.tick_params(axis='x', rotation=20)
        ax.set_ylabel('USDT')
        ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    plt.show()

    return hours_dt, hourly_cum_closed, equity_hourly
    
#endregion