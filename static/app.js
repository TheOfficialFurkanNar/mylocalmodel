let currentConversationId = null;
let isWorking = false;

// TTS state
let activeTTSSource = null;   // currently playing EventSource (null when idle)
let activeAudioCtx = null;    // Web AudioContext for the active playback session
let ttsNextStartTime = 0;     // scheduled start time for the next audio chunk

const messagesContainer = document.getElementById('messages');
const conversationList = document.getElementById('conversationList');
const conversationTitle = document.getElementById('activeConversationTitle');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const voiceSelect = document.getElementById('voiceSelect');

document.getElementById('newConversationButton').addEventListener('click', startNewConversation);

// ---------------------------------------------------------------------------
// Voice selector — populate from /tts/voices on startup
// ---------------------------------------------------------------------------

async function loadVoices() {
    try {
        const response = await fetch('/tts/voices');
        const data = await response.json();
        const voices = data.voices || [];
        const savedVoice = localStorage.getItem('localqwen-tts-voice') || 'af_heart';
        voiceSelect.replaceChildren();

        // Group by accent
        const groups = { US: [], UK: [] };
        voices.forEach((v) => { (groups[v.accent] || []).push(v); });

        for (const [accent, label] of [['US', 'American English'], ['UK', 'British English']]) {
            if (!groups[accent].length) continue;
            const optgroup = document.createElement('optgroup');
            optgroup.label = label;
            groups[accent].forEach((v) => {
                const opt = document.createElement('option');
                opt.value = v.id;
                opt.textContent = `${v.name} (${v.gender})`;
                opt.selected = v.id === savedVoice;
                optgroup.appendChild(opt);
            });
            voiceSelect.appendChild(optgroup);
        }
    } catch (err) {
        console.warn('Could not load TTS voices:', err);
        voiceSelect.innerHTML = '<option value="af_heart">Heart (F) – default</option>';
    }
}

voiceSelect.addEventListener('change', () => {
    localStorage.setItem('localqwen-tts-voice', voiceSelect.value);
    stopTTS(); // cancel any in-progress playback when voice changes
});

chatForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const question = messageInput.value.trim();
    if (!question || isWorking) return;

    addUserMessage(question);
    messageInput.value = '';
    const assistantId = `assistant-${Date.now()}`;
    renderAssistantShell(assistantId);
    addLoadingIndicator(assistantId);

    isWorking = true;
    sendButton.disabled = true;
    messageInput.disabled = true;
    try {
        await sendMessage(question, assistantId);
    } catch (error) {
        console.error('Chat error:', error);
        updateAssistantText(assistantId, `Error: ${error.message}`);
        removeLoadingIndicator(assistantId);
    } finally {
        isWorking = false;
        sendButton.disabled = false;
        messageInput.disabled = false;
        messageInput.focus();
    }
});

async function initializeHistory() {
    try {
        const response = await fetch('/conversations');
        const data = await response.json();
        const items = data.conversations || [];
        renderConversationList(items);

        const savedId = localStorage.getItem('localqwen-current-conversation');
        const selected = items.find((item) => item.id === savedId) || items[0];
        if (selected) await openConversation(selected.id);
    } catch (error) {
        console.error('Could not load conversation history:', error);
    }
}

async function sendMessage(question, assistantId) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000);
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: question, conversation_id: currentConversationId }),
            signal: controller.signal
        });
        if (!response.ok) {
            let detail = response.statusText;
            try {
                const errData = await response.json();
                if (errData.detail) detail = errData.detail;
                else if (errData.error) detail = errData.error;
            } catch (_) {}
            throw new Error(`HTTP ${response.status}: ${detail}`);
        }
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        currentConversationId = data.conversation_id;
        localStorage.setItem('localqwen-current-conversation', currentConversationId);
        if (data.metadata?.rag_matches?.length) {
            renderRAGCard(assistantId, data.metadata.rag_matches);
        }
        (data.metadata?.tool_calls || []).forEach((toolCall, index) => {
            renderToolCallCard(assistantId, index, toolCall);
        });
        updateAssistantText(assistantId, data.response || '');
        renderMetadataCard(assistantId, data.metadata || {});
        removeLoadingIndicator(assistantId);
        // Attach TTS speaker button — getText reads the rendered plain text at click time.
        attachSpeakerButton(assistantId, () => {
            return document.getElementById(`answer-${assistantId}`)?.innerText || data.response || '';
        });
        const cleanTitle = question.replace(/\s+/g, ' ').trim();
        conversationTitle.textContent = cleanTitle ? (cleanTitle.slice(0, 58) + (cleanTitle.length > 58 ? '…' : '')) : 'New chat';
        await refreshConversationList();
    } catch (error) {
        if (error.name === 'AbortError') throw new Error('Request timed out after 5 minutes');
        throw error;
    } finally {
        clearTimeout(timeoutId);
    }
}

async function refreshConversationList() {
    const response = await fetch('/conversations');
    const data = await response.json();
    renderConversationList(data.conversations || []);
}

function renderConversationList(conversations) {
    conversationList.replaceChildren();
    if (!conversations.length) {
        const empty = document.createElement('p');
        empty.className = 'history-empty';
        empty.textContent = 'Your chats will appear here.';
        conversationList.appendChild(empty);
        return;
    }

    conversations.forEach((conversation) => {
        const row = document.createElement('div');
        row.className = 'conversation-row';
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'conversation-item';
        if (conversation.id === currentConversationId) button.classList.add('active');
        const title = document.createElement('span');
        title.className = 'conversation-item-title';
        title.textContent = conversation.title || 'New chat';
        const detail = document.createElement('span');
        detail.className = 'conversation-item-detail';
        detail.textContent = `${formatDate(conversation.updated_at)} · ${Math.ceil(conversation.message_count / 2)} turns`;
        button.append(title, detail);
        button.addEventListener('click', () => openConversation(conversation.id));

        const deleteButton = document.createElement('button');
        deleteButton.type = 'button';
        deleteButton.className = 'conversation-delete';
        deleteButton.textContent = '×';
        deleteButton.setAttribute('aria-label', `Delete conversation: ${conversation.title || 'New chat'}`);
        deleteButton.title = 'Delete conversation';
        deleteButton.disabled = isWorking;
        deleteButton.addEventListener('click', () => deleteConversation(conversation));

        row.append(button, deleteButton);
        conversationList.appendChild(row);
    });
}

async function deleteConversation(conversation) {
    if (isWorking || !window.confirm(`Delete “${conversation.title || 'New chat'}”? This cannot be undone.`)) return;
    try {
        const response = await fetch(`/conversations/${encodeURIComponent(conversation.id)}`, { method: 'DELETE' });
        if (!response.ok) throw new Error(`Conversation could not be deleted (HTTP ${response.status})`);
        if (currentConversationId === conversation.id) startNewConversation();
        await refreshConversationList();
    } catch (error) {
        console.error('Could not delete conversation:', error);
        window.alert(error.message);
    }
}

async function openConversation(conversationId) {
    if (isWorking) return;
    try {
        const response = await fetch(`/conversations/${encodeURIComponent(conversationId)}`);
        if (!response.ok) throw new Error('Conversation could not be loaded');
        const conversation = await response.json();
        currentConversationId = conversation.id;
        localStorage.setItem('localqwen-current-conversation', currentConversationId);
        conversationTitle.textContent = conversation.title || 'New chat';
        messagesContainer.replaceChildren();

        for (const message of conversation.messages || []) {
            if (message.role === 'user') {
                addUserMessage(message.content, false);
            } else if (message.role === 'assistant') {
                const assistantId = message.id || `assistant-${Date.now()}-${Math.random()}`;
                renderAssistantShell(assistantId);
                const metadata = message.metadata || {};
                if (metadata.rag_matches?.length) {
                    renderRAGCard(assistantId, metadata.rag_matches);
                }
                (metadata.tool_calls || []).forEach((toolCall, index) => {
                    renderToolCallCard(assistantId, index, toolCall);
                });
                updateAssistantText(assistantId, message.content || '');
                renderMetadataCard(assistantId, metadata);
                // Attach TTS speaker button for reopened conversation messages.
                const capturedContent = message.content || '';
                attachSpeakerButton(assistantId, () => {
                    return document.getElementById(`answer-${assistantId}`)?.innerText || capturedContent;
                });
            }
        }
        await refreshConversationList();
        scrollToBottom();
    } catch (error) {
        console.error('Could not open conversation:', error);
    }
}

function startNewConversation() {
    if (isWorking) return;
    currentConversationId = null;
    localStorage.removeItem('localqwen-current-conversation');
    conversationTitle.textContent = 'New chat';
    messagesContainer.replaceChildren();
    renderConversationListFromCurrent();
    messageInput.focus();
}

async function renderConversationListFromCurrent() {
    try {
        await refreshConversationList();
    } catch (error) {
        console.error('Could not refresh conversation history:', error);
    }
}

function addUserMessage(content, scroll = true) {
    const wrapper = document.createElement('div');
    wrapper.className = 'message user';
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '👤';
    const text = document.createElement('div');
    text.className = 'message-content user-message-text';
    text.textContent = content;
    wrapper.append(avatar, text);
    messagesContainer.appendChild(wrapper);
    if (scroll) scrollToBottom();
}

function renderAssistantShell(assistantId) {
    const wrapper = document.createElement('div');
    wrapper.className = 'message assistant';
    wrapper.id = `message-${assistantId}`;
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';
    const content = document.createElement('div');
    content.className = 'message-content';
    content.id = `message-content-${assistantId}`;
    wrapper.append(avatar, content);
    messagesContainer.appendChild(wrapper);
}

// ---------------------------------------------------------------------------
// TTS — streaming playback via Web Audio API + SSE
// ---------------------------------------------------------------------------

function stopTTS() {
    if (activeTTSSource) {
        activeTTSSource.close();
        activeTTSSource = null;
    }
    if (activeAudioCtx) {
        activeAudioCtx.close().catch(() => {});
        activeAudioCtx = null;
    }
    ttsNextStartTime = 0;
    // Reset all speaker buttons to idle state
    document.querySelectorAll('.tts-btn.playing').forEach((btn) => {
        btn.textContent = '🔊';
        btn.classList.remove('playing');
        btn.title = 'Read aloud';
    });
}

/**
 * Inject a speaker button into the message content area.
 * getText: function that returns the plain-text content to synthesise.
 */
function attachSpeakerButton(assistantId, getText) {
    const content = document.getElementById(`message-content-${assistantId}`);
    if (!content) return;

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'tts-btn';
    btn.textContent = '🔊';
    btn.title = 'Read aloud';

    btn.addEventListener('click', async () => {
        // Toggle off if already playing.
        if (btn.classList.contains('playing')) {
            stopTTS();
            return;
        }
        // Stop any other ongoing playback first.
        stopTTS();

        const text = typeof getText === 'function' ? getText() : getText;
        if (!text || !text.trim()) return;

        const voice = voiceSelect.value || 'af_heart';

        btn.textContent = '⏳';
        btn.classList.add('playing');
        btn.title = 'Stop';

        try {
            const ctx = new AudioContext();
            activeAudioCtx = ctx;
            ttsNextStartTime = ctx.currentTime;

            const url = `/tts/stream?text=${encodeURIComponent(text)}&voice=${encodeURIComponent(voice)}`;
            const eventSource = new EventSource(url);
            activeTTSSource = eventSource;
            let firstChunk = true;

            eventSource.onmessage = async (e) => {
                if (!e.data) return;
                try {
                    // Decode base64 WAV blob sent per sentence chunk.
                    const binary = atob(e.data);
                    const bytes = new Uint8Array(binary.length);
                    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

                    const audioBuffer = await ctx.decodeAudioData(bytes.buffer.slice(0));

                    if (firstChunk) {
                        btn.textContent = '⏹';
                        firstChunk = false;
                    }

                    const source = ctx.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(ctx.destination);

                    // Schedule chunk immediately after the previous one ends.
                    const startAt = Math.max(ttsNextStartTime, ctx.currentTime + 0.01);
                    source.start(startAt);
                    ttsNextStartTime = startAt + audioBuffer.duration;
                } catch (decodeErr) {
                    console.warn('[TTS] Chunk decode error:', decodeErr);
                }
            };

            eventSource.addEventListener('done', () => {
                eventSource.close();
                activeTTSSource = null;
                // Reset button after the last chunk finishes playing.
                const delay = Math.max(0, (ttsNextStartTime - ctx.currentTime) * 1000 + 200);
                setTimeout(() => {
                    btn.textContent = '🔊';
                    btn.classList.remove('playing');
                    btn.title = 'Read aloud';
                }, delay);
            });

            eventSource.addEventListener('error', () => {
                stopTTS();
            });
        } catch (err) {
            console.error('[TTS] Playback error:', err);
            stopTTS();
        }
    });

    content.appendChild(btn);
}

function updateAssistantText(assistantId, value) {
    const content = document.getElementById(`message-content-${assistantId}`);
    if (!content) return;
    let answer = document.getElementById(`answer-${assistantId}`);
    if (!answer) {
        answer = document.createElement('div');
        answer.className = 'assistant-text';
        answer.id = `answer-${assistantId}`;
        content.appendChild(answer);
    }
    if (window.marked && window.DOMPurify) {
        try {
            const protectedMath = protectLatexDelimiters(value);
            const parseFn = typeof window.marked.parse === 'function' ? window.marked.parse.bind(window.marked) : window.marked;
            const markdown = parseFn(protectedMath.text, { gfm: true, breaks: true });
            answer.innerHTML = window.DOMPurify.sanitize(markdown, { USE_PROFILES: { html: true } });
            if (window.renderMathInElement) {
                restoreLatexDelimiters(answer, protectedMath.expressions);
                addInlineMathDelimiters(answer);
                window.renderMathInElement(answer, {
                    delimiters: [
                        { left: '$$', right: '$$', display: true },
                        { left: '\\[', right: '\\]', display: true },
                        { left: '\\(', right: '\\)', display: false },
                        { left: '$', right: '$', display: false }
                    ],
                    throwOnError: false
                });
            }
        } catch (renderError) {
            console.warn('Rich text rendering failed, falling back to plain text:', renderError);
            answer.textContent = value;
        }
    } else {
        answer.textContent = value;
    }
    scrollToBottom();
}

function protectLatexDelimiters(value) {
    const expressions = [];
    const delimiters = /\\\[([\s\S]*?)\\\]|\\\(([\s\S]*?)\\\)|\$\$([\s\S]*?)\$\$|\$([^\n$]+?)\$/g;
    const text = value.replace(delimiters, (expression, displayBracket, inlineParen, displayDollar, inlineDollar) => {
        const index = expressions.push(expression) - 1;
        return `LATEXPLACEHOLDER${index}TOKEN`;
    });
    return { text, expressions };
}

function restoreLatexDelimiters(root, expressions) {
    if (!expressions.length) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);
    const placeholderPattern = /LATEXPLACEHOLDER(\d+)TOKEN/g;
    for (const node of textNodes) {
        const text = node.nodeValue;
        const chunks = [];
        let cursor = 0;
        let match;
        while ((match = placeholderPattern.exec(text)) !== null) {
            chunks.push(document.createTextNode(text.slice(cursor, match.index)));
            chunks.push(document.createTextNode(expressions[Number(match[1])] || match[0]));
            cursor = match.index + match[0].length;
        }
        if (cursor > 0) {
            chunks.push(document.createTextNode(text.slice(cursor)));
            const fragment = document.createDocumentFragment();
            chunks.forEach((chunk) => fragment.appendChild(chunk));
            node.replaceWith(fragment);
        }
        placeholderPattern.lastIndex = 0;
    }
}

function addInlineMathDelimiters(root) {
    const groupedCommands = new Map([
        ['frac', 2], ['dfrac', 2], ['tfrac', 2], ['sqrt', 1],
        ['text', 1], ['mathrm', 1], ['mathbf', 1], ['mathit', 1],
        ['overline', 1], ['underline', 1]
    ]);
    const mathCommands = new Set([
        ...groupedCommands.keys(), 'pi', 'theta', 'alpha', 'beta', 'gamma', 'delta',
        'lambda', 'mu', 'sigma', 'omega', 'pm', 'mp', 'times', 'cdot', 'div',
        'neq', 'leq', 'geq', 'approx', 'partial', 'infty', 'sum', 'prod', 'int',
        'log', 'ln', 'sin', 'cos', 'tan', 'exp', 'lim', 'to'
    ]);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);

    for (const node of textNodes) {
        if (node.parentElement?.closest('code, pre, .katex') || /\$|\\\(|\\\[/.test(node.nodeValue)) continue;
        const text = node.nodeValue;
        const commandPattern = /\\([a-zA-Z]+)/g;
        const chunks = [];
        let cursor = 0;
        let match;
        while ((match = commandPattern.exec(text)) !== null) {
            const command = match[1];
            if (!mathCommands.has(command)) continue;
            let end = match.index + match[0].length;
            const argumentCount = groupedCommands.get(command) || 0;
            let valid = true;
            for (let argument = 0; argument < argumentCount; argument += 1) {
                while (/\s/.test(text[end] || '') && end < text.length) end += 1;
                if (command === 'sqrt' && argument === 0 && text[end] === '[') {
                    const optionalEnd = text.indexOf(']', end + 1);
                    if (optionalEnd >= 0) end = optionalEnd + 1;
                    while (/\s/.test(text[end] || '') && end < text.length) end += 1;
                }
                if (text[end] !== '{') {
                    valid = false;
                    break;
                }
                let depth = 0;
                let groupEnd = end;
                for (; groupEnd < text.length; groupEnd += 1) {
                    if (text[groupEnd] === '{' && text[groupEnd - 1] !== '\\') depth += 1;
                    if (text[groupEnd] === '}' && text[groupEnd - 1] !== '\\') depth -= 1;
                    if (depth === 0) break;
                }
                if (depth !== 0) {
                    valid = false;
                    break;
                }
                end = groupEnd + 1;
            }
            if (!valid || match.index < cursor) continue;
            chunks.push(document.createTextNode(text.slice(cursor, match.index)));
            chunks.push(document.createTextNode(`\\(${text.slice(match.index, end)}\\)`));
            cursor = end;
            commandPattern.lastIndex = end;
        }
        if (cursor > 0) {
            chunks.push(document.createTextNode(text.slice(cursor)));
            const fragment = document.createDocumentFragment();
            chunks.forEach((chunk) => fragment.appendChild(chunk));
            node.replaceWith(fragment);
        }
    }
}

function renderRAGCard(assistantId, matches) {
    const content = document.getElementById(`message-content-${assistantId}`);
    if (!content || !matches || !matches.length) return;
    const card = document.createElement('details');
    card.className = 'rag-card';
    const summary = document.createElement('summary');
    summary.textContent = `📚 Wikipedia QA Reference · ${matches.length} match${matches.length > 1 ? 'es' : ''}`;
    card.appendChild(summary);

    matches.forEach((match, idx) => {
        const item = document.createElement('div');
        item.className = 'rag-item';

        const q = document.createElement('div');
        q.className = 'rag-q';
        q.textContent = `Q: ${match.question || ''}`;

        const a = document.createElement('div');
        a.className = 'rag-a';
        a.textContent = `A: ${match.answer || ''}`;

        const score = document.createElement('div');
        score.className = 'rag-score';
        score.textContent = `Distance: ${match.distance ?? 'n/a'} · Similarity: ${match.similarity ?? 'n/a'}`;

        item.append(q, a, score);
        card.appendChild(item);
    });
    content.appendChild(card);
}

function renderToolCallCard(assistantId, index, toolCall) {
    const content = document.getElementById(`message-content-${assistantId}`);
    if (!content) return;
    const card = document.createElement('details');
    card.className = 'tool-card';
    const summary = document.createElement('summary');
    summary.textContent = `🧮 Calculator · ${toolCall.success === false ? 'error' : `${toolCall.elapsed_ms ?? 0} ms`}`;
    card.appendChild(summary);
    appendMetadataLine(card, 'Expression', toolCall.expression || '');
    appendMetadataLine(card, toolCall.success === false ? 'Error' : 'Result', toolCall.error || toolCall.result || '');
    content.appendChild(card);
}

function renderMetadataCard(assistantId, metadata) {
    const content = document.getElementById(`message-content-${assistantId}`);
    if (!content || !metadata.timestamp_utc) return;
    const details = document.createElement('details');
    details.className = 'turn-metadata';
    const summary = document.createElement('summary');
    summary.textContent = 'Turn details';
    details.appendChild(summary);

    appendMetadataLine(details, 'Timestamp', formatDate(metadata.timestamp_utc, true));
    appendMetadataLine(details, 'Duration', `${metadata.duration_seconds ?? 0} s`);
    appendMetadataLine(details, 'Model', metadata.model_name || '');
    appendMetadataLine(details, 'Calculator used', metadata.tool_used ? 'Yes' : 'No');
    appendMetadataLine(details, 'RAG used', metadata.rag_used ? `Yes (${metadata.rag_matches?.length || 0} match(es))` : 'No');
    appendMetadataLine(details, 'Context / new token limit', `${metadata.max_length} / ${metadata.max_new_tokens}`);
    appendMetadataLine(details, 'Best-of-N', metadata.best_of_n);
    if (metadata.generation) {
        appendMetadataLine(
            details,
            'Sampling',
            `temp ${metadata.generation.temperature}; calibration ${metadata.generation.calibration_temperature}; top-k ${metadata.generation.top_k}; top-p ${metadata.generation.top_p}`
        );
    }
    if (metadata.accepted_candidate) {
        const accepted = metadata.accepted_candidate;
        appendMetadataLine(
            details,
            'Accepted candidate',
            `#${accepted.index} (${accepted.stage}; score ${formatScore(accepted.score, 4)})`
        );
    }
    (metadata.candidate_runs || []).forEach((run) => {
        const scores = (run.candidates || [])
            .map((candidate) => `${candidate.selected ? '*' : ''}#${candidate.index} ${formatScore(candidate.score, 3)}`)
            .join(' · ');
        appendMetadataLine(
            details,
            `Candidates · ${run.stage}`,
            `${run.prompt_tokens} prompt tokens; accepted #${run.accepted_candidate} / ${run.candidate_count}: ${scores}`
        );
    });
    content.appendChild(details);
}

function formatScore(value, digits) {
    return Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
}

function appendMetadataLine(parent, label, value) {
    const row = document.createElement('div');
    row.className = 'metadata-line';
    const labelElement = document.createElement('span');
    labelElement.className = 'metadata-label';
    labelElement.textContent = label;
    const valueElement = document.createElement('span');
    valueElement.className = 'metadata-value';
    valueElement.textContent = value ?? '';
    row.append(labelElement, valueElement);
    parent.appendChild(row);
}

function addLoadingIndicator(assistantId) {
    const content = document.getElementById(`message-content-${assistantId}`);
    const loading = document.createElement('div');
    loading.className = 'loading';
    loading.id = `loading-${assistantId}`;
    loading.innerHTML = '<div class="spinner"></div><span>Generating candidates and checking calculations…</span>';
    content.appendChild(loading);
}

function removeLoadingIndicator(assistantId) {
    document.getElementById(`loading-${assistantId}`)?.remove();
}

function formatDate(value, includeTime = false) {
    if (!value) return '';
    const date = new Date(value);
    return new Intl.DateTimeFormat(undefined, includeTime
        ? { dateStyle: 'medium', timeStyle: 'short' }
        : { dateStyle: 'medium' }).format(date);
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

initializeHistory();
loadVoices();
