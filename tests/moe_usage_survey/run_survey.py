#!/usr/bin/env python3
"""
MoE usage survey harness.

Runs a fixed list of 106 diverse real-world prompts (programming across many
languages, CS fundamentals, code review/debug/explain with real embedded
code, daily-life/general questions, and one resume-improvement prompt)
against the real production GLM-5.3 model (mirror_model_path stays enabled,
freeze_usage_stats disabled) so RamSafetyManager/GpuTopology's routing
histogram (.moestream_usage-equivalent, redirected to a dedicated
`moestream_usage_survey` file so this doesn't pollute the live production
usage file) accumulates real per-(layer,expert) hit counts across a
realistic prompt mix. Purpose: identify genuinely infrequently-used experts
as real candidates for a future IQ1_S conversion, instead of guessing.

Each prompt is written to prompts/NNN_slug.txt (audit trail of exactly what
was sent) and its full raw stdout (prompt echo + generated response +
engine banner) is written to responses/NNN_slug.txt, plus a copy of that
run's own debug log to responses/NNN_slug.log. index.tsv accumulates one
row per prompt with status/timing so progress can be checked at any point
without waiting for the whole batch.

Runs strictly sequentially (one moestream process at a time -- this engine
owns the GPUs exclusively). Designed to be safe to re-run: if index.tsv
already has a COMPLETED row for a given id, that prompt is skipped, so an
interrupted batch can just be restarted.
"""
import subprocess
import sys
import os
import re
import time
import shutil
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MOESTREAM = Path(os.environ.get("MOESTREAM_SURVEY_BINARY",
                               str(REPO / "build" / "moestream")))
CONFIG = Path(os.environ.get("MOESTREAM_SURVEY_CONFIG",
                            str(HERE / "survey_config.toml")))
PROMPTS_DIR = Path(os.environ.get("MOESTREAM_SURVEY_PROMPTS_DIR",
                                  str(HERE / "prompts")))
RESPONSES_DIR = Path(os.environ.get("MOESTREAM_SURVEY_RESPONSES_DIR",
                                   str(HERE / "responses")))
INDEX_TSV = Path(os.environ.get("MOESTREAM_SURVEY_INDEX",
                               str(HERE / "index.tsv")))
SURVEY_LOG = Path(os.environ.get("MOESTREAM_SURVEY_LOG",
                                str(HERE / "survey_debug.log")))
USAGE_FILE = Path(os.environ.get("MOESTREAM_SURVEY_USAGE_FILE",
                               str(HERE / "moestream_usage_survey")))
ENGINE_LOG_PATH = os.environ.get("MOESTREAM_SURVEY_ENGINE_LOG_PATH", "")
USE_GDB = os.environ.get("MOESTREAM_SURVEY_GDB", "0") == "1"

NGEN = int(os.environ.get("MOESTREAM_SURVEY_NGEN", "128"))
FORCE_RERUN = os.environ.get("MOESTREAM_SURVEY_FORCE_RERUN", "0") == "1"
PERSISTENT = os.environ.get("MOESTREAM_SURVEY_PERSISTENT", "0") == "1"
PER_PROMPT_TIMEOUT_S = 900  # generous margin above the ~200-250s expected


def engine_env():
    """Return the child environment, including an explicit CUDA mask if set."""
    env = os.environ.copy()
    visible = os.environ.get("MOESTREAM_SURVEY_CUDA_VISIBLE_DEVICES")
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
    return env


def wrap_engine_command(command):
    """Run the engine under GDB when the sandbox blocks direct CUDA ioctls."""
    if not USE_GDB:
        return command
    return ["gdb", "-q", "-batch", "-ex", "run", "--args", *command]

# ---------------------------------------------------------------------------
# Full prompt list: (category, slug, prompt_text)
# ---------------------------------------------------------------------------
PROMPTS = []


def add(category, slug, text):
    PROMPTS.append((category, slug, text.strip()))


# ---- Python (7: 5 write + 2 review) ----
add("python", "csv_average", "Write a Python function that reads a CSV file and returns the average of a numeric column.")
add("python", "lru_cache", "Implement a simple LRU cache in Python without using functools.")
add("python", "fibonacci_generator", "Write a Python generator that yields Fibonacci numbers indefinitely.")
add("python", "binary_search_tree", "Write a Python class implementing a basic binary search tree with insert and in-order traversal.")
add("python", "asyncio_fetch", "Write a Python script using asyncio to fetch multiple URLs concurrently.")
add("python", "review_order_total", """Review this Python function and point out any bugs or bad practices:
```python
def process_orders(orders):
    total = 0
    for order in orders:
        total = total + order['amount']
        if order['status'] == 'refunded':
            total = total - order['amount']
    average = total / len(orders)
    return average
```""")
add("python", "explain_list_comprehension_candidate", """What does this Python code do, and is there a more idiomatic way to write it?
```python
result = []
for i in range(len(data)):
    if data[i] % 2 == 0:
        result.append(data[i] * 2)
```""")

# ---- Shell/Bash (6: 5 write + 1 review) ----
add("bash", "find_large_files", "Write a bash script that finds all files larger than 100MB in a directory tree and lists them sorted by size.")
add("bash", "restart_process", "Write a bash function that checks if a given process name is running and restarts it if not.")
add("bash", "backup_tar", "Write a bash script that backs up a directory to a timestamped tar.gz archive.")
add("bash", "getopts_parsing", "Write a bash script that parses command-line arguments with getopts, supporting -v and --file.")
add("bash", "disk_usage_alert", "Write a bash script that monitors disk usage and sends an alert if usage exceeds 90%.")
add("bash", "review_log_cleanup", """Review this bash script for potential bugs:
```bash
#!/bin/bash
for f in $(ls *.log); do
  if [ $f -gt 1000000 ]; then
    rm $f
  fi
done
```""")

# ---- JavaScript & TypeScript (7: 5 write + 2 review) ----
add("js_ts", "debounce", "Write a JavaScript function that debounces another function by a given delay.")
add("js_ts", "retry_fetch", "Write a JavaScript async function that retries a fetch() call up to 3 times.")
add("js_ts", "deep_clone_generics", "Write a TypeScript function using generics to deep-clone an object.")
add("js_ts", "react_counter", "Write a TypeScript React functional component with a counter using useState.")
add("js_ts", "promise_sleep", "Write a JavaScript Promise-based sleep function and use it in an async loop.")
add("js_ts", "review_age_bug", """Review this JavaScript function for bugs:
```javascript
function getUserAge(user) {
  if (user.birthYear = 2000) {
    return new Date().getFullYear() - user.birthYear;
  }
  return null;
}
```""")
add("js_ts", "explain_merge_types", """Explain what this TypeScript code does and whether the types are safe:
```typescript
function merge(a: any, b: any) {
  return { ...a, ...b };
}
```""")

# ---- Go (6: 4 write + 2 review) ----
add("go", "http_health_server", "Write a Go HTTP server with a single /health endpoint returning JSON.")
add("go", "stack_struct", "Write a Go struct with methods implementing a simple stack.")
add("go", "waitgroup_goroutines", "Write a Go function that uses a WaitGroup to wait for multiple goroutines to finish.")
add("go", "pretty_print_json_stdin", "Write a Go program that reads JSON from stdin and pretty-prints it.")
add("go", "review_counter_race", """Review this Go code for concurrency bugs:
```go
func main() {
    counter := 0
    var wg sync.WaitGroup
    for i := 0; i < 100; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            counter++
        }()
    }
    wg.Wait()
    fmt.Println(counter)
}
```""")
add("go", "review_error_handling", """What does this Go code do, and is there a bug in the error handling?
```go
func readConfig(path string) (*Config, error) {
    data, err := os.ReadFile(path)
    var cfg Config
    json.Unmarshal(data, &cfg)
    if err != nil {
        return nil, err
    }
    return &cfg, nil
}
```""")

# ---- C & C++ (7: 5 write + 2 review) ----
add("c_cpp", "quicksort_c", "Write a C function implementing quicksort on an integer array.")
add("c_cpp", "template_binary_search", "Write a C++ template function for a generic binary search over a sorted vector.")
add("c_cpp", "thread_safe_queue", "Write a C++ class implementing a thread-safe queue using std::mutex and std::condition_variable.")
add("c_cpp", "raii_file_handle", "Write a C++ RAII wrapper class for a raw file handle.")
add("c_cpp", "producer_consumer_pthreads", "Write a C program demonstrating a basic producer-consumer pattern with pthreads.")
add("c_cpp", "review_dangling_pointer", """Review this C code for memory safety issues:
```c
char* build_greeting(const char* name) {
    char buffer[20];
    sprintf(buffer, "Hello, %s!", name);
    return buffer;
}
```""")
add("c_cpp", "review_delete_vs_delete_array", """Review this C++ code for potential issues:
```cpp
class Buffer {
public:
    Buffer(int size) { data = new int[size]; }
    ~Buffer() { delete data; }
private:
    int* data;
};
```""")

# ---- JSON (4) ----
add("json", "user_schema", 'Write a JSON schema describing a "user" object with id, name, email, and an array of roles.')
add("json", "product_catalog", "Convert this description into JSON: a product catalog entry with id, name, price, tags, and nested dimensions (width/height/depth).")
add("json", "paginated_api_response", "Write a JSON example representing a paginated API response with metadata (page, total, per_page) and a data array.")
add("json", "server_config", "Write a JSON config file example for a simple web server (port, host, routes array, logging level).")

# ---- HTML (8: 7 write + 1 review) ----
add("html", "form_validation", "Write an HTML form with fields for name, email, and password, including basic client-side validation attributes.")
add("html", "semantic_skeleton", "Write a semantic HTML5 page skeleton with header, nav, main, article, and footer.")
add("html", "employee_table", "Write an HTML table with a caption, thead, and tbody displaying a list of employees.")
add("html", "responsive_srcset_image", "Write an HTML page embedding a responsive image with srcset for different screen sizes.")
add("html", "accessible_modal", "Write an accessible HTML button with aria-label and a modal dialog structure.")
add("html", "select_radio_checkbox_form", "Write an HTML5 form with a <select> dropdown, radio buttons, and a checkbox group, using semantic <fieldset>/<legend>.")
add("html", "details_summary_faq", "Write an HTML page using <details>/<summary> for a collapsible FAQ section.")
add("html", "review_accessibility", """Review this HTML for accessibility issues:
```html
<div onclick="submitForm()">Submit</div>
<img src="logo.png">
<input type="text">
```""")

# ---- CSS (6: 5 write + 1 review) ----
add("css", "flexbox_navbar", "Write CSS using Flexbox to create a responsive navbar that collapses items on small screens.")
add("css", "grid_cards", "Write CSS Grid layout code for a 3-column responsive card layout that becomes 1 column on mobile.")
add("css", "box_model", "Write CSS demonstrating the box model and how box-sizing: border-box changes it.")
add("css", "custom_properties_theme", "Write CSS using custom properties (variables) to implement a light/dark theme toggle.")
add("css", "specificity_explain", "Explain CSS specificity and write an example showing how !important and specificity interact.")
add("css", "review_specificity_maintainability", """Review this CSS for specificity or maintainability issues:
```css
#header .nav ul li a {
  color: red !important;
}
.nav a {
  color: blue;
}
```""")

# ---- SQL (6: 5 write + 1 review) ----
add("sql", "top5_customers", "Write a SQL query to find the top 5 customers by total order amount.")
add("sql", "create_orders_table", 'Write a SQL CREATE TABLE statement for an "orders" table with proper foreign keys and constraints.')
add("sql", "window_function_rank", "Write a SQL query using a window function to rank employees by salary within each department.")
add("sql", "duplicate_rows", "Write a SQL query to find duplicate rows in a table based on a subset of columns.")
add("sql", "update_counter_transaction", "Write a SQL UPDATE statement that increments a counter column, with a WHERE clause and a transaction.")
add("sql", "review_implicit_join", """Review this SQL query for performance and correctness issues:
```sql
SELECT * FROM orders o, customers c
WHERE o.customer_id = c.id
AND o.created_at > '2023-01-01'
ORDER BY o.amount;
```""")

# ---- TCP/IP & Networking (5) ----
add("networking", "tcp_handshake", "Explain the TCP three-way handshake and what happens if the final ACK is lost.")
add("networking", "tcp_vs_udp", "Explain the difference between TCP and UDP, and give a real example of when you'd choose UDP.")
add("networking", "osi_model_webpage", "Explain what happens at each layer of the OSI model when you load a webpage.")
add("networking", "dns_resolution", "Explain how DNS resolution works, including caching and recursive vs iterative queries.")
add("networking", "subnet_mask", "Explain what a subnet mask does and how to calculate the number of usable hosts in a /26 network.")

# ---- Operating Systems (5) ----
add("os", "process_vs_thread", "Explain the difference between a process and a thread, and the tradeoffs of each.")
add("os", "virtual_memory_paging", "Explain how virtual memory and paging work, including what a page fault is.")
add("os", "deadlock", "Explain deadlock, the four necessary conditions for it, and one strategy to prevent it.")
add("os", "preemptive_vs_cooperative", "Explain the difference between preemptive and cooperative multitasking.")
add("os", "context_switch", "Explain what a context switch is and why it has overhead.")

# ---- Linux (6) ----
add("linux", "hard_vs_symbolic_link", "Explain the difference between a hard link and a symbolic link in Linux.")
add("linux", "file_permissions", "Explain how Linux file permissions (rwx, owner/group/other) work, and what chmod 754 means.")
add("linux", "systemd_units", "Explain what systemd units are and the difference between a service and a timer unit.")
add("linux", "page_cache", 'Explain how the Linux page cache works and why "free" memory can look low but not be a problem.')
add("linux", "zombie_vs_orphan", "Explain the difference between a zombie process and an orphan process in Linux.")
add("linux", "strace_ltrace", "Explain what strace and ltrace are used for, and give an example use case.")

# ---- Daily / general questions (10, verbatim) ----
add("daily", "meaning_of_life", "What is the meaning of life?")
add("daily", "learn_new_language", "How can I acquire a new language?")
add("daily", "ai_vs_ml", "What distinguishes AI from machine learning?")
add("daily", "recommended_books", "What are the most recommended books to read?")
add("daily", "improve_writing", "How can I enhance my writing proficiency?")
add("daily", "future_of_tech", "What does the future of technology hold?")
add("daily", "meditation_benefits", "What are the advantages of meditation?")
add("daily", "world_history", "What is the world's history?")
add("daily", "boost_productivity", "How can I boost my productivity?")
add("daily", "significance_of_art", "What is the significance of art?")

# ---- Resume (1) ----
add("resume", "improve_summary", """Here is a summary from my resume — please help me improve it:

"Software engineer with 4 years experience. Worked on backend systems using Python and Go. Familiar with databases and cloud. Good team player and fast learner. Looking for new opportunities in a challenging environment."
""")

# ---- Debugging (3, real errors) ----
add("debugging", "keyerror_traceback", """I'm getting this error, what's wrong and how do I fix it?
```
Traceback (most recent call last):
  File "app.py", line 12, in <module>
    result = process(data)
  File "app.py", line 7, in process
    return data['value'] / count
KeyError: 'value'
```
Here's the relevant code:
```python
def process(items):
    count = len(items)
    data = items[0]
    return data['value'] / count
```""")
add("debugging", "unhandled_promise_rejection", """This Node.js code throws `UnhandledPromiseRejectionWarning`, what's the issue?
```javascript
async function loadUser(id) {
  const res = await fetch(`/api/users/${id}`);
  return res.json();
}
loadUser(123).then(user => console.log(user.name));
```""")
add("debugging", "cpp_intermittent_segfault", """This C++ code segfaults intermittently, what's the likely cause?
```cpp
std::vector<int> getEvens(std::vector<int>& nums) {
    std::vector<int> result;
    for (auto it = nums.begin(); it != nums.end(); ++it) {
        if (*it % 2 == 0) {
            nums.erase(it);
        }
    }
    return result;
}
```""")

# ---- Testing (1) ----
add("testing", "pytest_palindrome", """Write pytest unit tests for this function:
```python
def is_palindrome(s):
    s = s.lower().replace(' ', '')
    return s == s[::-1]
```""")

# ---- Regex (2) ----
add("regex", "email_validation", "Write a regex that validates an email address.")
add("regex", "hashtag_extraction", "Write a regex to extract all hashtags from a block of text.")

# ---- Git (3) ----
add("git", "squash_commits", "How do I squash my last 3 commits into one?")
add("git", "gitignore_python_node", "Write a good .gitignore for a Python + Node.js monorepo.")
add("git", "resolve_merge_conflict_cli", "How do I resolve a merge conflict in a single file using the command line?")

# ---- Docker / YAML (3) ----
add("docker_yaml", "dockerfile_flask", "Write a Dockerfile for a Python Flask app with a requirements.txt.")
add("docker_yaml", "docker_compose_web_db_cache", "Write a docker-compose.yml for a web app + Postgres + Redis.")
add("docker_yaml", "github_actions_pytest", "Write a GitHub Actions YAML workflow that runs pytest on every push to main.")

# ---- Algorithms (3) ----
add("algorithms", "mergesort_vs_quicksort", "Explain the time and space complexity of merge sort vs quicksort.")
add("algorithms", "reverse_linked_list", "Reverse a linked list in place — explain the approach, then write the code.")
add("algorithms", "bfs_vs_dfs", "Explain the difference between BFS and DFS, with a use case for each.")

# ---- Design patterns / README (2) ----
add("design_patterns", "observer_pattern", "Explain the Observer pattern with a short code example.")
add("design_patterns", "readme_cli_tool", "Write a README.md for an open-source CLI tool (sections: install, usage, example, license).")

# ---- General (5) ----
add("general", "decline_meeting_email", "Write a professional email declining a meeting invite due to a scheduling conflict.")
add("general", "eli5_quantum_computing", "Explain quantum computing like I'm five years old.")
add("general", "python_vs_js_beginner", "Should I learn Python or JavaScript first as a beginner, and why?")
add("general", "summarize_distributed_systems", """Summarize this paragraph in two sentences:

"Distributed systems are collections of independent computers that appear to their users as a single coherent system. They communicate and coordinate their actions by passing messages over a network, which introduces challenges not present in single-machine systems: partial failures, network delays, and the impossibility of perfectly synchronized clocks. Designers of distributed systems must reason carefully about consistency, availability, and partition tolerance, since the CAP theorem shows that a system cannot simultaneously guarantee all three during a network partition. Real-world systems typically make deliberate tradeoffs among these properties depending on their specific requirements, favoring strong consistency for financial transactions and higher availability for content delivery, for example."
""")
add("general", "fix_grammar_paragraph", """Fix the grammar in this paragraph:

"Me and my team has been working on this project since three months. We are hopeing to finish it by next week, but there is alot of bugs that we still need to fixed. Its been a challenging experience but we learn alot from it."
""")

assert len(PROMPTS) == 106, f"expected 106 prompts, got {len(PROMPTS)}"


def slugify(category, slug):
    return f"{category}__{slug}"


def run_one(idx, category, slug, text):
    name = f"{idx:03d}_{slugify(category, slug)}"
    prompt_file = PROMPTS_DIR / f"{name}.txt"
    response_file = RESPONSES_DIR / f"{name}.txt"
    log_file = RESPONSES_DIR / f"{name}.log"

    prompt_file.write_text(text + "\n")

    # main.cpp's REPL reads exactly one getline() per turn -- a real
    # embedded newline in `text` (any multi-line prompt: code review, the
    # resume/summarize/grammar-fix prompts) would otherwise be split into
    # multiple separate, wrong turns. Escape real newlines to the literal
    # two-char "\n" sequence main.cpp now un-escapes right after reading
    # (see that fix's own comment). Only affects what's piped to stdin --
    # prompt_file/response_file above and below still record the real,
    # readable multi-line text for auditing.
    stdin_text = text.replace("\n", "\\n")

    t0 = time.time()
    status = "OK"
    stdout_text = ""
    try:
        command = [
            str(MOESTREAM),
            "--config",
            str(CONFIG),
            "--ngen",
            str(NGEN),
            "--collect-usage-stats",
        ]
        if ENGINE_LOG_PATH:
            command += ["--log-path", ENGINE_LOG_PATH]
        survey_pin_file = os.environ.get("MOESTREAM_SURVEY_PIN_FILE")
        if survey_pin_file:
            command += ["--pin-file", survey_pin_file]
        command = wrap_engine_command(command)
        proc = subprocess.run(
            command,
            input=stdin_text + "\n",
            capture_output=True,
            text=True,
            timeout=PER_PROMPT_TIMEOUT_S,
            cwd=str(REPO),
            env=engine_env(),
        )
        stdout_text = proc.stdout
        if proc.returncode != 0:
            status = f"NONZERO_EXIT_{proc.returncode}"
    except subprocess.TimeoutExpired as e:
        status = "TIMEOUT"
        stdout_text = (e.stdout or "") if isinstance(e.stdout, str) else ""
    wall_s = time.time() - t0

    # Extract the generated response between "Output:" and "[Generation
    # Complete]" for a quick-read convenience section; the FULL raw stdout
    # is always saved too, so this parse being imperfect never loses data.
    m = re.search(r"Output:\s*(.*?)\n\[Generation Complete\]", stdout_text, re.S)
    extracted = m.group(1).strip() if m else "(could not parse — see full raw output below)"

    error_count = stdout_text.count("[ERROR]")
    if SURVEY_LOG.exists():
        try:
            log_text = SURVEY_LOG.read_text(errors="replace")
            error_count += log_text.count("[ERROR]")
            log_file.write_text(log_text)
        except Exception:
            pass

    response_file.write_text(
        f"=== PROMPT ({category}/{slug}) ===\n{text}\n\n"
        f"=== EXTRACTED RESPONSE ===\n{extracted}\n\n"
        f"=== STATUS: {status}  wall_s={wall_s:.1f}  error_lines={error_count} ===\n\n"
        f"=== FULL RAW STDOUT ===\n{stdout_text}\n"
    )

    return status, wall_s, error_count


def run_persistent(items):
    """Run all pending prompts through one long-lived engine process."""
    if not items:
        return

    prompt_lines = []
    for idx, category, slug, text in items:
        name = f"{idx:03d}_{slugify(category, slug)}"
        (PROMPTS_DIR / f"{name}.txt").write_text(text + "\n")
        prompt_lines.append(text.replace("\n", "\\n"))

    command = [
        str(MOESTREAM),
        "--config",
        str(CONFIG),
        "--ngen",
        str(NGEN),
        "--collect-usage-stats",
    ]
    if ENGINE_LOG_PATH:
        command += ["--log-path", ENGINE_LOG_PATH]
    survey_pin_file = os.environ.get("MOESTREAM_SURVEY_PIN_FILE")
    if survey_pin_file:
        command += ["--pin-file", survey_pin_file]
    command = wrap_engine_command(command)

    print(f"Persistent survey: one engine process for {len(items)} prompts", flush=True)
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(REPO),
        env=engine_env(),
    )

    try:
        proc.stdin.write("\n".join(prompt_lines) + "\n:q\n")
        proc.stdin.close()
    except BrokenPipeError:
        pass

    marker = "[Generation Complete]"
    pending_output = ""
    completed = 0
    while True:
        chunk = proc.stdout.read(4096)
        if not chunk:
            break
        pending_output += chunk
        while marker in pending_output and completed < len(items):
            segment, pending_output = pending_output.split(marker, 1)
            idx, category, slug, text = items[completed]
            name = f"{idx:03d}_{slugify(category, slug)}"
            response_file = RESPONSES_DIR / f"{name}.txt"
            log_file = RESPONSES_DIR / f"{name}.log"
            segment_with_marker = segment + "\n" + marker + "\n"
            match = re.search(
                r"Output:\s*(.*?)\n\[Generation Complete\]",
                segment_with_marker,
                re.S,
            )
            extracted = (
                match.group(1).strip()
                if match
                else "(could not parse — see full raw output below)"
            )
            error_count = segment.count("[ERROR]")
            response_file.write_text(
                f"=== PROMPT ({category}/{slug}) ===\n{text}\n\n"
                f"=== EXTRACTED RESPONSE ===\n{extracted}\n\n"
                f"=== STATUS: OK  wall_s=persistent ===\n\n"
                f"=== FULL RAW STDOUT ===\n{segment_with_marker}\n"
            )
            if SURVEY_LOG.exists():
                try:
                    log_file.write_text(SURVEY_LOG.read_text(errors="replace"))
                except Exception:
                    pass
            with open(INDEX_TSV, "a") as index:
                index.write(
                    f"{idx}\t{category}\t{slug}\tOK\t-1\t"
                    f"{error_count}\tDONE\n"
                )
            completed += 1
            print(
                f"[{idx:03d}/{len(PROMPTS)}] {category}/{slug}: "
                "OK (persistent process)",
                flush=True,
            )

    rc = proc.wait()
    if completed != len(items):
        if pending_output:
            failure_log = RESPONSES_DIR / "persistent_engine_failure.log"
            failure_log.write_text(pending_output)
            print(
                f"Persistent engine output saved to {failure_log}:\n"
                f"{pending_output[-8000:]}",
                file=sys.stderr,
            )
        print(
            f"Persistent survey ended with exit={rc}; "
            f"completed {completed}/{len(items)} prompts",
            file=sys.stderr,
        )


def already_done(idx):
    if not INDEX_TSV.exists():
        return False
    with open(INDEX_TSV) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] == str(idx) and parts[-1] == "DONE":
                return True
    return False


def main():
    global CONFIG, NGEN

    parser = argparse.ArgumentParser(description="Run the MoE usage prompt survey")
    parser.add_argument(
        "--config",
        default=str(CONFIG),
        help="engine TOML config (default: %(default)s)",
    )
    parser.add_argument(
        "--ngen",
        type=int,
        default=NGEN,
        help="maximum generated tokens per prompt (default: %(default)s)",
    )
    parser.add_argument(
        "--persistent",
        action="store_true",
        default=PERSISTENT,
        help="reuse one engine process for all prompts",
    )
    args = parser.parse_args()

    CONFIG = Path(args.config).resolve()
    NGEN = args.ngen

    if not MOESTREAM.exists():
        print(f"ERROR: {MOESTREAM} not found — build it first (cmake --build build --target moestream)", file=sys.stderr)
        sys.exit(1)

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_TSV.parent.mkdir(parents=True, exist_ok=True)
    if not INDEX_TSV.exists():
        INDEX_TSV.write_text("id\tcategory\tslug\tstatus\twall_s\terror_lines\tmarker\n")

    print(f"Survey: {len(PROMPTS)} prompts, ngen={NGEN}, "
          f"timeout={PER_PROMPT_TIMEOUT_S}s/prompt, "
          f"force_rerun={FORCE_RERUN}, persistent={args.persistent}, "
          f"config={CONFIG}, usage_file={USAGE_FILE}")

    pending = [
        (idx, category, slug, text)
        for idx, (category, slug, text) in enumerate(PROMPTS, start=1)
        if FORCE_RERUN or not already_done(idx)
    ]
    if args.persistent:
        run_persistent(pending)
        pending = []

    if args.persistent:
        prompt_iter = []
    else:
        prompt_iter = pending
    for idx, category, slug, text in prompt_iter:
        print(f"[{idx:03d}/{len(PROMPTS)}] {category}/{slug}: running...", flush=True)
        status, wall_s, error_count = run_one(idx, category, slug, text)
        with open(INDEX_TSV, "a") as f:
            f.write(f"{idx}\t{category}\t{slug}\t{status}\t{wall_s:.1f}\t{error_count}\tDONE\n")
        print(f"[{idx:03d}/{len(PROMPTS)}] {category}/{slug}: {status} in {wall_s:.1f}s, {error_count} error line(s)", flush=True)

    print("Survey complete.")

    # Final analysis: rank (layer,expert) by hit count from the accumulated
    # dedicated survey usage file.
    if USAGE_FILE.exists():
        entries = []
        with open(USAGE_FILE) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3:
                    layer, expert, count = int(parts[0]), int(parts[1]), int(parts[2])
                    entries.append((layer, expert, count))
        entries.sort(key=lambda e: e[2])
        print(f"\n.moestream_usage_survey: {len(entries)} (layer,expert) pairs recorded.")
        print("Coldest 20:")
        for layer, expert, count in entries[:20]:
            print(f"  layer={layer} expert={expert} count={count}")
        print("Hottest 10:")
        for layer, expert, count in entries[-10:]:
            print(f"  layer={layer} expert={expert} count={count}")


if __name__ == "__main__":
    main()
