var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key2, value) => key2 in obj ? __defProp(obj, key2, { enumerable: true, configurable: true, writable: true, value }) : obj[key2] = value;
var __publicField = (obj, key2, value) => {
  __defNormalProp(obj, typeof key2 !== "symbol" ? key2 + "" : key2, value);
  return value;
};
(function polyfill() {
  const relList = document.createElement("link").relList;
  if (relList && relList.supports && relList.supports("modulepreload")) {
    return;
  }
  for (const link2 of document.querySelectorAll('link[rel="modulepreload"]')) {
    processPreload(link2);
  }
  new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.type !== "childList") {
        continue;
      }
      for (const node of mutation.addedNodes) {
        if (node.tagName === "LINK" && node.rel === "modulepreload")
          processPreload(node);
      }
    }
  }).observe(document, { childList: true, subtree: true });
  function getFetchOpts(link2) {
    const fetchOpts = {};
    if (link2.integrity)
      fetchOpts.integrity = link2.integrity;
    if (link2.referrerPolicy)
      fetchOpts.referrerPolicy = link2.referrerPolicy;
    if (link2.crossOrigin === "use-credentials")
      fetchOpts.credentials = "include";
    else if (link2.crossOrigin === "anonymous")
      fetchOpts.credentials = "omit";
    else
      fetchOpts.credentials = "same-origin";
    return fetchOpts;
  }
  function processPreload(link2) {
    if (link2.ep)
      return;
    link2.ep = true;
    const fetchOpts = getFetchOpts(link2);
    fetch(link2.href, fetchOpts);
  }
})();
function noop$2() {
}
function assign(tar, src) {
  for (const k in src)
    tar[k] = src[k];
  return (
    /** @type {T & S} */
    tar
  );
}
function is_promise(value) {
  return !!value && (typeof value === "object" || typeof value === "function") && typeof /** @type {any} */
  value.then === "function";
}
function run$1(fn) {
  return fn();
}
function blank_object() {
  return /* @__PURE__ */ Object.create(null);
}
function run_all(fns) {
  fns.forEach(run$1);
}
function is_function(thing) {
  return typeof thing === "function";
}
function safe_not_equal(a, b) {
  return a != a ? b == b : a !== b || a && typeof a === "object" || typeof a === "function";
}
function is_empty(obj) {
  return Object.keys(obj).length === 0;
}
function subscribe(store2, ...callbacks) {
  if (store2 == null) {
    for (const callback of callbacks) {
      callback(void 0);
    }
    return noop$2;
  }
  const unsub = store2.subscribe(...callbacks);
  return unsub.unsubscribe ? () => unsub.unsubscribe() : unsub;
}
function get_store_value(store2) {
  let value;
  subscribe(store2, (_) => value = _)();
  return value;
}
function component_subscribe(component, store2, callback) {
  component.$$.on_destroy.push(subscribe(store2, callback));
}
function create_slot(definition, ctx, $$scope, fn) {
  if (definition) {
    const slot_ctx = get_slot_context(definition, ctx, $$scope, fn);
    return definition[0](slot_ctx);
  }
}
function get_slot_context(definition, ctx, $$scope, fn) {
  return definition[1] && fn ? assign($$scope.ctx.slice(), definition[1](fn(ctx))) : $$scope.ctx;
}
function get_slot_changes(definition, $$scope, dirty, fn) {
  if (definition[2] && fn) {
    const lets = definition[2](fn(dirty));
    if ($$scope.dirty === void 0) {
      return lets;
    }
    if (typeof lets === "object") {
      const merged = [];
      const len = Math.max($$scope.dirty.length, lets.length);
      for (let i2 = 0; i2 < len; i2 += 1) {
        merged[i2] = $$scope.dirty[i2] | lets[i2];
      }
      return merged;
    }
    return $$scope.dirty | lets;
  }
  return $$scope.dirty;
}
function update_slot_base(slot, slot_definition, ctx, $$scope, slot_changes, get_slot_context_fn) {
  if (slot_changes) {
    const slot_context = get_slot_context(slot_definition, ctx, $$scope, get_slot_context_fn);
    slot.p(slot_context, slot_changes);
  }
}
function get_all_dirty_from_scope($$scope) {
  if ($$scope.ctx.length > 32) {
    const dirty = [];
    const length = $$scope.ctx.length / 32;
    for (let i2 = 0; i2 < length; i2++) {
      dirty[i2] = -1;
    }
    return dirty;
  }
  return -1;
}
function exclude_internal_props(props) {
  const result = {};
  for (const k in props)
    if (k[0] !== "$")
      result[k] = props[k];
  return result;
}
function compute_rest_props(props, keys) {
  const rest = {};
  keys = new Set(keys);
  for (const k in props)
    if (!keys.has(k) && k[0] !== "$")
      rest[k] = props[k];
  return rest;
}
function set_store_value(store2, ret, value) {
  store2.set(value);
  return ret;
}
function action_destroyer(action_result) {
  return action_result && is_function(action_result.destroy) ? action_result.destroy : noop$2;
}
const contenteditable_truthy_values = ["", true, 1, "true", "contenteditable"];
function append$1(target, node) {
  target.appendChild(node);
}
function insert(target, node, anchor2) {
  target.insertBefore(node, anchor2 || null);
}
function detach(node) {
  if (node.parentNode) {
    node.parentNode.removeChild(node);
  }
}
function destroy_each(iterations, detaching) {
  for (let i2 = 0; i2 < iterations.length; i2 += 1) {
    if (iterations[i2])
      iterations[i2].d(detaching);
  }
}
function element(name2) {
  return document.createElement(name2);
}
function svg_element(name2) {
  return document.createElementNS("http://www.w3.org/2000/svg", name2);
}
function text(data) {
  return document.createTextNode(data);
}
function space() {
  return text(" ");
}
function empty$1() {
  return text("");
}
function listen(node, event, handler, options) {
  node.addEventListener(event, handler, options);
  return () => node.removeEventListener(event, handler, options);
}
function attr(node, attribute, value) {
  if (value == null)
    node.removeAttribute(attribute);
  else if (node.getAttribute(attribute) !== value)
    node.setAttribute(attribute, value);
}
const always_set_through_set_attribute = ["width", "height"];
function set_attributes(node, attributes) {
  const descriptors = Object.getOwnPropertyDescriptors(node.__proto__);
  for (const key2 in attributes) {
    if (attributes[key2] == null) {
      node.removeAttribute(key2);
    } else if (key2 === "style") {
      node.style.cssText = attributes[key2];
    } else if (key2 === "__value") {
      node.value = node[key2] = attributes[key2];
    } else if (descriptors[key2] && descriptors[key2].set && always_set_through_set_attribute.indexOf(key2) === -1) {
      node[key2] = attributes[key2];
    } else {
      attr(node, key2, attributes[key2]);
    }
  }
}
function set_svg_attributes(node, attributes) {
  for (const key2 in attributes) {
    attr(node, key2, attributes[key2]);
  }
}
function children$1(element2) {
  return Array.from(element2.childNodes);
}
function set_data(text2, data) {
  data = "" + data;
  if (text2.data === data)
    return;
  text2.data = /** @type {string} */
  data;
}
function set_data_contenteditable(text2, data) {
  data = "" + data;
  if (text2.wholeText === data)
    return;
  text2.data = /** @type {string} */
  data;
}
function set_data_maybe_contenteditable(text2, data, attr_value) {
  if (~contenteditable_truthy_values.indexOf(attr_value)) {
    set_data_contenteditable(text2, data);
  } else {
    set_data(text2, data);
  }
}
function toggle_class(element2, name2, toggle2) {
  element2.classList.toggle(name2, !!toggle2);
}
function custom_event(type, detail, { bubbles = false, cancelable = false } = {}) {
  return new CustomEvent(type, { detail, bubbles, cancelable });
}
function construct_svelte_component(component, props) {
  return new component(props);
}
let current_component;
function set_current_component(component) {
  current_component = component;
}
function get_current_component() {
  if (!current_component)
    throw new Error("Function called outside component initialization");
  return current_component;
}
function onMount(fn) {
  get_current_component().$$.on_mount.push(fn);
}
function afterUpdate(fn) {
  get_current_component().$$.after_update.push(fn);
}
function onDestroy(fn) {
  get_current_component().$$.on_destroy.push(fn);
}
function createEventDispatcher() {
  const component = get_current_component();
  return (type, detail, { cancelable = false } = {}) => {
    const callbacks = component.$$.callbacks[type];
    if (callbacks) {
      const event = custom_event(
        /** @type {string} */
        type,
        detail,
        { cancelable }
      );
      callbacks.slice().forEach((fn) => {
        fn.call(component, event);
      });
      return !event.defaultPrevented;
    }
    return true;
  };
}
const dirty_components = [];
const binding_callbacks = [];
let render_callbacks = [];
const flush_callbacks = [];
const resolved_promise = /* @__PURE__ */ Promise.resolve();
let update_scheduled = false;
function schedule_update() {
  if (!update_scheduled) {
    update_scheduled = true;
    resolved_promise.then(flush);
  }
}
function tick() {
  schedule_update();
  return resolved_promise;
}
function add_render_callback(fn) {
  render_callbacks.push(fn);
}
function add_flush_callback(fn) {
  flush_callbacks.push(fn);
}
const seen_callbacks = /* @__PURE__ */ new Set();
let flushidx = 0;
function flush() {
  if (flushidx !== 0) {
    return;
  }
  const saved_component = current_component;
  do {
    try {
      while (flushidx < dirty_components.length) {
        const component = dirty_components[flushidx];
        flushidx++;
        set_current_component(component);
        update$1(component.$$);
      }
    } catch (e) {
      dirty_components.length = 0;
      flushidx = 0;
      throw e;
    }
    set_current_component(null);
    dirty_components.length = 0;
    flushidx = 0;
    while (binding_callbacks.length)
      binding_callbacks.pop()();
    for (let i2 = 0; i2 < render_callbacks.length; i2 += 1) {
      const callback = render_callbacks[i2];
      if (!seen_callbacks.has(callback)) {
        seen_callbacks.add(callback);
        callback();
      }
    }
    render_callbacks.length = 0;
  } while (dirty_components.length);
  while (flush_callbacks.length) {
    flush_callbacks.pop()();
  }
  update_scheduled = false;
  seen_callbacks.clear();
  set_current_component(saved_component);
}
function update$1($$) {
  if ($$.fragment !== null) {
    $$.update();
    run_all($$.before_update);
    const dirty = $$.dirty;
    $$.dirty = [-1];
    $$.fragment && $$.fragment.p($$.ctx, dirty);
    $$.after_update.forEach(add_render_callback);
  }
}
function flush_render_callbacks(fns) {
  const filtered = [];
  const targets = [];
  render_callbacks.forEach((c) => fns.indexOf(c) === -1 ? filtered.push(c) : targets.push(c));
  targets.forEach((c) => c());
  render_callbacks = filtered;
}
const outroing = /* @__PURE__ */ new Set();
let outros;
function group_outros() {
  outros = {
    r: 0,
    c: [],
    p: outros
    // parent group
  };
}
function check_outros() {
  if (!outros.r) {
    run_all(outros.c);
  }
  outros = outros.p;
}
function transition_in(block2, local) {
  if (block2 && block2.i) {
    outroing.delete(block2);
    block2.i(local);
  }
}
function transition_out(block2, local, detach2, callback) {
  if (block2 && block2.o) {
    if (outroing.has(block2))
      return;
    outroing.add(block2);
    outros.c.push(() => {
      outroing.delete(block2);
      if (callback) {
        if (detach2)
          block2.d(1);
        callback();
      }
    });
    block2.o(local);
  } else if (callback) {
    callback();
  }
}
function handle_promise(promise, info) {
  const token = info.token = {};
  function update2(type, index, key2, value) {
    if (info.token !== token)
      return;
    info.resolved = value;
    let child_ctx = info.ctx;
    if (key2 !== void 0) {
      child_ctx = child_ctx.slice();
      child_ctx[key2] = value;
    }
    const block2 = type && (info.current = type)(child_ctx);
    let needs_flush = false;
    if (info.block) {
      if (info.blocks) {
        info.blocks.forEach((block3, i2) => {
          if (i2 !== index && block3) {
            group_outros();
            transition_out(block3, 1, 1, () => {
              if (info.blocks[i2] === block3) {
                info.blocks[i2] = null;
              }
            });
            check_outros();
          }
        });
      } else {
        info.block.d(1);
      }
      block2.c();
      transition_in(block2, 1);
      block2.m(info.mount(), info.anchor);
      needs_flush = true;
    }
    info.block = block2;
    if (info.blocks)
      info.blocks[index] = block2;
    if (needs_flush) {
      flush();
    }
  }
  if (is_promise(promise)) {
    const current_component2 = get_current_component();
    promise.then(
      (value) => {
        set_current_component(current_component2);
        update2(info.then, 1, info.value, value);
        set_current_component(null);
      },
      (error) => {
        set_current_component(current_component2);
        update2(info.catch, 2, info.error, error);
        set_current_component(null);
        if (!info.hasCatch) {
          throw error;
        }
      }
    );
    if (info.current !== info.pending) {
      update2(info.pending, 0);
      return true;
    }
  } else {
    if (info.current !== info.then) {
      update2(info.then, 1, info.value, promise);
      return true;
    }
    info.resolved = /** @type {T} */
    promise;
  }
}
function update_await_block_branch(info, ctx, dirty) {
  const child_ctx = ctx.slice();
  const { resolved } = info;
  if (info.current === info.then) {
    child_ctx[info.value] = resolved;
  }
  if (info.current === info.catch) {
    child_ctx[info.error] = resolved;
  }
  info.block.p(child_ctx, dirty);
}
function ensure_array_like(array_like_or_iterator) {
  return (array_like_or_iterator == null ? void 0 : array_like_or_iterator.length) !== void 0 ? array_like_or_iterator : Array.from(array_like_or_iterator);
}
function outro_and_destroy_block(block2, lookup) {
  transition_out(block2, 1, 1, () => {
    lookup.delete(block2.key);
  });
}
function update_keyed_each(old_blocks, dirty, get_key, dynamic, ctx, list2, lookup, node, destroy, create_each_block2, next2, get_context) {
  let o = old_blocks.length;
  let n = list2.length;
  let i2 = o;
  const old_indexes = {};
  while (i2--)
    old_indexes[old_blocks[i2].key] = i2;
  const new_blocks = [];
  const new_lookup = /* @__PURE__ */ new Map();
  const deltas = /* @__PURE__ */ new Map();
  const updates = [];
  i2 = n;
  while (i2--) {
    const child_ctx = get_context(ctx, list2, i2);
    const key2 = get_key(child_ctx);
    let block2 = lookup.get(key2);
    if (!block2) {
      block2 = create_each_block2(key2, child_ctx);
      block2.c();
    } else if (dynamic) {
      updates.push(() => block2.p(child_ctx, dirty));
    }
    new_lookup.set(key2, new_blocks[i2] = block2);
    if (key2 in old_indexes)
      deltas.set(key2, Math.abs(i2 - old_indexes[key2]));
  }
  const will_move = /* @__PURE__ */ new Set();
  const did_move = /* @__PURE__ */ new Set();
  function insert2(block2) {
    transition_in(block2, 1);
    block2.m(node, next2);
    lookup.set(block2.key, block2);
    next2 = block2.first;
    n--;
  }
  while (o && n) {
    const new_block = new_blocks[n - 1];
    const old_block = old_blocks[o - 1];
    const new_key = new_block.key;
    const old_key = old_block.key;
    if (new_block === old_block) {
      next2 = new_block.first;
      o--;
      n--;
    } else if (!new_lookup.has(old_key)) {
      destroy(old_block, lookup);
      o--;
    } else if (!lookup.has(new_key) || will_move.has(new_key)) {
      insert2(new_block);
    } else if (did_move.has(old_key)) {
      o--;
    } else if (deltas.get(new_key) > deltas.get(old_key)) {
      did_move.add(new_key);
      insert2(new_block);
    } else {
      will_move.add(old_key);
      o--;
    }
  }
  while (o--) {
    const old_block = old_blocks[o];
    if (!new_lookup.has(old_block.key))
      destroy(old_block, lookup);
  }
  while (n)
    insert2(new_blocks[n - 1]);
  run_all(updates);
  return new_blocks;
}
function get_spread_update(levels, updates) {
  const update2 = {};
  const to_null_out = {};
  const accounted_for = { $$scope: 1 };
  let i2 = levels.length;
  while (i2--) {
    const o = levels[i2];
    const n = updates[i2];
    if (n) {
      for (const key2 in o) {
        if (!(key2 in n))
          to_null_out[key2] = 1;
      }
      for (const key2 in n) {
        if (!accounted_for[key2]) {
          update2[key2] = n[key2];
          accounted_for[key2] = 1;
        }
      }
      levels[i2] = n;
    } else {
      for (const key2 in o) {
        accounted_for[key2] = 1;
      }
    }
  }
  for (const key2 in to_null_out) {
    if (!(key2 in update2))
      update2[key2] = void 0;
  }
  return update2;
}
function bind(component, name2, callback) {
  const index = component.$$.props[name2];
  if (index !== void 0) {
    component.$$.bound[index] = callback;
    callback(component.$$.ctx[index]);
  }
}
function create_component(block2) {
  block2 && block2.c();
}
function mount_component(component, target, anchor2) {
  const { fragment, after_update } = component.$$;
  fragment && fragment.m(target, anchor2);
  add_render_callback(() => {
    const new_on_destroy = component.$$.on_mount.map(run$1).filter(is_function);
    if (component.$$.on_destroy) {
      component.$$.on_destroy.push(...new_on_destroy);
    } else {
      run_all(new_on_destroy);
    }
    component.$$.on_mount = [];
  });
  after_update.forEach(add_render_callback);
}
function destroy_component(component, detaching) {
  const $$ = component.$$;
  if ($$.fragment !== null) {
    flush_render_callbacks($$.after_update);
    run_all($$.on_destroy);
    $$.fragment && $$.fragment.d(detaching);
    $$.on_destroy = $$.fragment = null;
    $$.ctx = [];
  }
}
function make_dirty(component, i2) {
  if (component.$$.dirty[0] === -1) {
    dirty_components.push(component);
    schedule_update();
    component.$$.dirty.fill(0);
  }
  component.$$.dirty[i2 / 31 | 0] |= 1 << i2 % 31;
}
function init$1(component, options, instance2, create_fragment2, not_equal, props, append_styles = null, dirty = [-1]) {
  const parent_component = current_component;
  set_current_component(component);
  const $$ = component.$$ = {
    fragment: null,
    ctx: [],
    // state
    props,
    update: noop$2,
    not_equal,
    bound: blank_object(),
    // lifecycle
    on_mount: [],
    on_destroy: [],
    on_disconnect: [],
    before_update: [],
    after_update: [],
    context: new Map(options.context || (parent_component ? parent_component.$$.context : [])),
    // everything else
    callbacks: blank_object(),
    dirty,
    skip_bound: false,
    root: options.target || parent_component.$$.root
  };
  append_styles && append_styles($$.root);
  let ready = false;
  $$.ctx = instance2 ? instance2(component, options.props || {}, (i2, ret, ...rest) => {
    const value = rest.length ? rest[0] : ret;
    if ($$.ctx && not_equal($$.ctx[i2], $$.ctx[i2] = value)) {
      if (!$$.skip_bound && $$.bound[i2])
        $$.bound[i2](value);
      if (ready)
        make_dirty(component, i2);
    }
    return ret;
  }) : [];
  $$.update();
  ready = true;
  run_all($$.before_update);
  $$.fragment = create_fragment2 ? create_fragment2($$.ctx) : false;
  if (options.target) {
    if (options.hydrate) {
      const nodes = children$1(options.target);
      $$.fragment && $$.fragment.l(nodes);
      nodes.forEach(detach);
    } else {
      $$.fragment && $$.fragment.c();
    }
    if (options.intro)
      transition_in(component.$$.fragment);
    mount_component(component, options.target, options.anchor);
    flush();
  }
  set_current_component(parent_component);
}
class SvelteComponent {
  constructor() {
    /**
     * ### PRIVATE API
     *
     * Do not use, may change at any time
     *
     * @type {any}
     */
    __publicField(this, "$$");
    /**
     * ### PRIVATE API
     *
     * Do not use, may change at any time
     *
     * @type {any}
     */
    __publicField(this, "$$set");
  }
  /** @returns {void} */
  $destroy() {
    destroy_component(this, 1);
    this.$destroy = noop$2;
  }
  /**
   * @template {Extract<keyof Events, string>} K
   * @param {K} type
   * @param {((e: Events[K]) => void) | null | undefined} callback
   * @returns {() => void}
   */
  $on(type, callback) {
    if (!is_function(callback)) {
      return noop$2;
    }
    const callbacks = this.$$.callbacks[type] || (this.$$.callbacks[type] = []);
    callbacks.push(callback);
    return () => {
      const index = callbacks.indexOf(callback);
      if (index !== -1)
        callbacks.splice(index, 1);
    };
  }
  /**
   * @param {Partial<Props>} props
   * @returns {void}
   */
  $set(props) {
    if (this.$$set && !is_empty(props)) {
      this.$$.skip_bound = true;
      this.$$set(props);
      this.$$.skip_bound = false;
    }
  }
}
const PUBLIC_VERSION = "4";
if (typeof window !== "undefined")
  (window.__svelte || (window.__svelte = { v: /* @__PURE__ */ new Set() })).v.add(PUBLIC_VERSION);
const subscriber_queue = [];
function readable(value, start2) {
  return {
    subscribe: writable(value, start2).subscribe
  };
}
function writable(value, start2 = noop$2) {
  let stop2;
  const subscribers = /* @__PURE__ */ new Set();
  function set2(new_value) {
    if (safe_not_equal(value, new_value)) {
      value = new_value;
      if (stop2) {
        const run_queue = !subscriber_queue.length;
        for (const subscriber of subscribers) {
          subscriber[1]();
          subscriber_queue.push(subscriber, value);
        }
        if (run_queue) {
          for (let i2 = 0; i2 < subscriber_queue.length; i2 += 2) {
            subscriber_queue[i2][0](subscriber_queue[i2 + 1]);
          }
          subscriber_queue.length = 0;
        }
      }
    }
  }
  function update2(fn) {
    set2(fn(value));
  }
  function subscribe2(run2, invalidate = noop$2) {
    const subscriber = [run2, invalidate];
    subscribers.add(subscriber);
    if (subscribers.size === 1) {
      stop2 = start2(set2, update2) || noop$2;
    }
    run2(value);
    return () => {
      subscribers.delete(subscriber);
      if (subscribers.size === 0 && stop2) {
        stop2();
        stop2 = null;
      }
    };
  }
  return { set: set2, update: update2, subscribe: subscribe2 };
}
function derived(stores, fn, initial_value) {
  const single = !Array.isArray(stores);
  const stores_array = single ? [stores] : stores;
  if (!stores_array.every(Boolean)) {
    throw new Error("derived() expects stores as input, got a falsy value");
  }
  const auto = fn.length < 2;
  return readable(initial_value, (set2, update2) => {
    let started = false;
    const values = [];
    let pending = 0;
    let cleanup = noop$2;
    const sync2 = () => {
      if (pending) {
        return;
      }
      cleanup();
      const result = fn(single ? values[0] : values, set2, update2);
      if (auto) {
        set2(result);
      } else {
        cleanup = is_function(result) ? result : noop$2;
      }
    };
    const unsubscribers = stores_array.map(
      (store2, i2) => subscribe(
        store2,
        (value) => {
          values[i2] = value;
          pending &= ~(1 << i2);
          if (started) {
            sync2();
          }
        },
        () => {
          pending |= 1 << i2;
        }
      )
    );
    started = true;
    sync2();
    return function stop2() {
      run_all(unsubscribers);
      cleanup();
      started = false;
    };
  });
}
function readonly(store2) {
  return {
    subscribe: store2.subscribe.bind(store2)
  };
}
class ExperimentMetadata {
  constructor(name2, id2, ensembles, responses, parameters) {
    __publicField(this, "name");
    __publicField(this, "id");
    __publicField(this, "ensembles");
    __publicField(this, "responses");
    __publicField(this, "parameters");
    this.name = name2;
    this.id = id2;
    this.ensembles = ensembles;
    this.responses = responses;
    this.parameters = parameters;
  }
  get numEnsembles() {
    return Object.keys(this.ensembles).length;
  }
  static FromObject({
    name: name2,
    id: id2,
    ensembles,
    responses,
    parameters
  }) {
    return new ExperimentMetadata(name2, id2, ensembles, responses, parameters);
  }
  availableKeywords() {
    const summaryKeys = this.responses.summary.keys;
    const hasHistoryKeys = new Set(summaryKeys.filter((k) => k.endsWith("H")).map((k) => k.slice(0, -1)));
    const summaryKeyInfos = summaryKeys.filter((k) => !k.endsWith("H")).map((k) => ({
      key: k,
      kind: "summary",
      hasHistory: hasHistoryKeys.has(k),
      hasObservations: false
      // TODO
    }));
    const paramKeysInfos = Object.values(this.parameters).flatMap((v) => v.transfer_function_definitions.map((n) => ({
      key: n.name,
      kind: "parameter",
      hasHistory: false,
      hasObservations: false
    })));
    return [...summaryKeyInfos, ...paramKeysInfos];
  }
  getKeywordInfo(keyword) {
    return this.availableKeywords().find((k) => k.key === keyword);
  }
  ensembleIdToAlias(ensembleId) {
    const { iteration } = this.ensembles[ensembleId];
    return `ensemble:${iteration === 0 ? "first" : iteration === this.numEnsembles - 1 ? "last" : i}`;
  }
  ensemblesByAliasOrId() {
    const mapping = {};
    const allEnsembles = Object.values(this.ensembles);
    allEnsembles.forEach((ens, i2) => {
      if (i2 === 0)
        mapping["first"] = ens;
      if (i2 === allEnsembles.length - 1)
        mapping["last"] = ens;
      mapping[ens.iteration] = ens;
      mapping[ens.id] = ens;
    });
    return mapping;
  }
  ensembleAliasToId(ensembleAlias) {
    return this.ensemblesByAliasOrId()[ensembleAlias].id;
  }
  eachEnsemble(f) {
    Object.keys(this.ensembles).forEach((ensembleId, i2) => {
      const ensembleAlias = this.ensembleIdToAlias(ensembleId);
      f(ensembleId, ensembleAlias, i2);
    });
  }
  sortedEnsembles() {
    return Object.values(this.ensembles).sort((a, b) => a.iteration - b.iteration);
  }
}
const urlParams$1 = new URLSearchParams(window.location.search);
const serverURL$1 = decodeURIComponent(urlParams$1.get("serverURL") || "http://localhost:8001");
const lag = async (t) => new Promise((resolve) => setTimeout(resolve, t));
const minLagMs = 0;
const createQueryString = (obj) => {
  const urlParams2 = new URLSearchParams();
  Object.entries(obj).forEach(([k, v]) => urlParams2.append(k, v.toString()));
  return urlParams2.toString();
};
const _summaryCache = {};
const fetchSummary = async (query) => {
  await lag(minLagMs);
  const queryString = createQueryString(query);
  if (!(queryString in _summaryCache)) {
    const response = await fetch(`${serverURL$1}/api/summary_chart_data?${queryString}`);
    const responseJSON = await response.json();
    _summaryCache[queryString] = responseJSON;
  }
  return _summaryCache[queryString];
};
const getLoadedSummary = (query) => {
  const queryString = createQueryString(query);
  if (!(queryString in _summaryCache))
    throw new ReferenceError(
      `Expected summary to be loaded for: ${queryString}.Loaded summaries: ${Object.keys(_summaryCache)}`
    );
  return _summaryCache[queryString];
};
let _experimentsMetadataCache = void 0;
const fetchExperiments = async (forceRefresh = false) => {
  await lag(minLagMs);
  if (!_experimentsMetadataCache || forceRefresh) {
    const response = await fetch(`${serverURL$1}/api/experiments`);
    const responseJSON = await response.json();
    const parsedResponseJSON = {};
    Object.entries(responseJSON).forEach(([k, v]) => parsedResponseJSON[k] = ExperimentMetadata.FromObject(v));
    _experimentsMetadataCache = parsedResponseJSON;
  }
  return _experimentsMetadataCache;
};
const getLoadedExperiments = () => _experimentsMetadataCache;
const _parametersCache = {};
const fetchParameter = async (query) => {
  await lag(minLagMs);
  const queryString = createQueryString(query);
  if (!(queryString in _parametersCache)) {
    const response = await fetch(`${serverURL$1}/api/parameter_chart_data?${queryString}`);
    const responseJSON = await response.json();
    _parametersCache[queryString] = responseJSON;
  }
  return _parametersCache[queryString];
};
const getLoadedParameter = (query) => {
  const queryString = createQueryString(query);
  if (!(queryString in _parametersCache))
    throw new ReferenceError(
      `Expected parameter to be loaded for: ${queryString}.Loaded parameters: ${Object.keys(_parametersCache)}`
    );
  return _parametersCache[createQueryString(query)];
};
const urlParams = new URLSearchParams(window.location.search);
const serverURL = decodeURIComponent(urlParams.get("serverURL") || ".");
const DefaultPlotterState = {
  serverURL,
  style: {
    // Global style
    "ensemble:first": {
      "stroke-width": "2px"
    },
    "ensemble:last": {
      "stroke-width": "2px"
    }
  },
  charts: [
    {
      kind: "summary",
      chart: "line",
      query: {
        ensembles: ["first", "last"],
        keyword: "FOPR",
        experiment: "auto"
      }
    },
    {
      kind: "parameter",
      chart: "ridgelines",
      query: {
        ensembles: ["first", "last"],
        keyword: "OP1_PERSISTENCE",
        experiment: "auto"
      },
      style: {
        // Global style
        "*": { opacity: 0.7 },
        "ensemble:first": {
          stroke: "none",
          "stroke-width": "2px"
        },
        "ensemble:last": {
          stroke: "none",
          "stroke-width": "2px"
        }
      }
    }
  ]
};
const serializedState = decodeURIComponent(urlParams.get("serializedState") || "");
let useState;
if (serializedState !== "") {
  useState = JSON.parse(serializedState);
} else {
  useState = DefaultPlotterState;
}
const ensureStoreIsSyncedWithExperiments = async () => {
  const experimentsMetadata = await fetchExperiments();
  const allExperiments = Object.keys(experimentsMetadata);
  useState.charts.forEach((spec) => {
    if (spec.query.experiment === "auto")
      spec.query.experiment = allExperiments[0];
    if (spec.kind === "summary") {
      const availableKeywords = experimentsMetadata[spec.query.experiment].responses.summary.keys;
      if (!availableKeywords.includes(spec.query.keyword)) {
        spec.query.keyword = availableKeywords[0];
      }
    } else if (spec.kind === "parameter") {
      const paramMeta = experimentsMetadata[spec.query.experiment].parameters;
      const availableParameters = [
        ...Object.entries(paramMeta).reduce((namesList, entry) => {
          const [, param] = entry;
          const paramNames = param.transfer_function_definitions.map((d) => d.name);
          return namesList.concat(paramNames);
        }, [])
      ];
      if (!availableParameters.includes(spec.query.keyword)) {
        spec.query.keyword = availableParameters[0];
      }
    }
  });
};
const plotterStore = writable(useState);
function ascending$1(a, b) {
  return a == null || b == null ? NaN : a < b ? -1 : a > b ? 1 : a >= b ? 0 : NaN;
}
function descending(a, b) {
  return a == null || b == null ? NaN : b < a ? -1 : b > a ? 1 : b >= a ? 0 : NaN;
}
function bisector(f) {
  let compare1, compare2, delta;
  if (f.length !== 2) {
    compare1 = ascending$1;
    compare2 = (d, x2) => ascending$1(f(d), x2);
    delta = (d, x2) => f(d) - x2;
  } else {
    compare1 = f === ascending$1 || f === descending ? f : zero$1;
    compare2 = f;
    delta = f;
  }
  function left2(a, x2, lo = 0, hi = a.length) {
    if (lo < hi) {
      if (compare1(x2, x2) !== 0)
        return hi;
      do {
        const mid = lo + hi >>> 1;
        if (compare2(a[mid], x2) < 0)
          lo = mid + 1;
        else
          hi = mid;
      } while (lo < hi);
    }
    return lo;
  }
  function right2(a, x2, lo = 0, hi = a.length) {
    if (lo < hi) {
      if (compare1(x2, x2) !== 0)
        return hi;
      do {
        const mid = lo + hi >>> 1;
        if (compare2(a[mid], x2) <= 0)
          lo = mid + 1;
        else
          hi = mid;
      } while (lo < hi);
    }
    return lo;
  }
  function center2(a, x2, lo = 0, hi = a.length) {
    const i2 = left2(a, x2, lo, hi - 1);
    return i2 > lo && delta(a[i2 - 1], x2) > -delta(a[i2], x2) ? i2 - 1 : i2;
  }
  return { left: left2, center: center2, right: right2 };
}
function zero$1() {
  return 0;
}
function number$2(x2) {
  return x2 === null ? NaN : +x2;
}
const ascendingBisect = bisector(ascending$1);
const bisectRight = ascendingBisect.right;
bisector(number$2).center;
const bisect = bisectRight;
function extent(values, valueof) {
  let min2;
  let max2;
  if (valueof === void 0) {
    for (const value of values) {
      if (value != null) {
        if (min2 === void 0) {
          if (value >= value)
            min2 = max2 = value;
        } else {
          if (min2 > value)
            min2 = value;
          if (max2 < value)
            max2 = value;
        }
      }
    }
  } else {
    let index = -1;
    for (let value of values) {
      if ((value = valueof(value, ++index, values)) != null) {
        if (min2 === void 0) {
          if (value >= value)
            min2 = max2 = value;
        } else {
          if (min2 > value)
            min2 = value;
          if (max2 < value)
            max2 = value;
        }
      }
    }
  }
  return [min2, max2];
}
class InternMap extends Map {
  constructor(entries, key2 = keyof) {
    super();
    Object.defineProperties(this, { _intern: { value: /* @__PURE__ */ new Map() }, _key: { value: key2 } });
    if (entries != null)
      for (const [key3, value] of entries)
        this.set(key3, value);
  }
  get(key2) {
    return super.get(intern_get(this, key2));
  }
  has(key2) {
    return super.has(intern_get(this, key2));
  }
  set(key2, value) {
    return super.set(intern_set(this, key2), value);
  }
  delete(key2) {
    return super.delete(intern_delete(this, key2));
  }
}
function intern_get({ _intern, _key }, value) {
  const key2 = _key(value);
  return _intern.has(key2) ? _intern.get(key2) : value;
}
function intern_set({ _intern, _key }, value) {
  const key2 = _key(value);
  if (_intern.has(key2))
    return _intern.get(key2);
  _intern.set(key2, value);
  return value;
}
function intern_delete({ _intern, _key }, value) {
  const key2 = _key(value);
  if (_intern.has(key2)) {
    value = _intern.get(key2);
    _intern.delete(key2);
  }
  return value;
}
function keyof(value) {
  return value !== null && typeof value === "object" ? value.valueOf() : value;
}
const e10 = Math.sqrt(50), e5 = Math.sqrt(10), e2 = Math.sqrt(2);
function tickSpec(start2, stop2, count) {
  const step = (stop2 - start2) / Math.max(0, count), power2 = Math.floor(Math.log10(step)), error = step / Math.pow(10, power2), factor = error >= e10 ? 10 : error >= e5 ? 5 : error >= e2 ? 2 : 1;
  let i1, i2, inc;
  if (power2 < 0) {
    inc = Math.pow(10, -power2) / factor;
    i1 = Math.round(start2 * inc);
    i2 = Math.round(stop2 * inc);
    if (i1 / inc < start2)
      ++i1;
    if (i2 / inc > stop2)
      --i2;
    inc = -inc;
  } else {
    inc = Math.pow(10, power2) * factor;
    i1 = Math.round(start2 / inc);
    i2 = Math.round(stop2 / inc);
    if (i1 * inc < start2)
      ++i1;
    if (i2 * inc > stop2)
      --i2;
  }
  if (i2 < i1 && 0.5 <= count && count < 2)
    return tickSpec(start2, stop2, count * 2);
  return [i1, i2, inc];
}
function ticks(start2, stop2, count) {
  stop2 = +stop2, start2 = +start2, count = +count;
  if (!(count > 0))
    return [];
  if (start2 === stop2)
    return [start2];
  const reverse = stop2 < start2, [i1, i2, inc] = reverse ? tickSpec(stop2, start2, count) : tickSpec(start2, stop2, count);
  if (!(i2 >= i1))
    return [];
  const n = i2 - i1 + 1, ticks2 = new Array(n);
  if (reverse) {
    if (inc < 0)
      for (let i3 = 0; i3 < n; ++i3)
        ticks2[i3] = (i2 - i3) / -inc;
    else
      for (let i3 = 0; i3 < n; ++i3)
        ticks2[i3] = (i2 - i3) * inc;
  } else {
    if (inc < 0)
      for (let i3 = 0; i3 < n; ++i3)
        ticks2[i3] = (i1 + i3) / -inc;
    else
      for (let i3 = 0; i3 < n; ++i3)
        ticks2[i3] = (i1 + i3) * inc;
  }
  return ticks2;
}
function tickIncrement(start2, stop2, count) {
  stop2 = +stop2, start2 = +start2, count = +count;
  return tickSpec(start2, stop2, count)[2];
}
function tickStep(start2, stop2, count) {
  stop2 = +stop2, start2 = +start2, count = +count;
  const reverse = stop2 < start2, inc = reverse ? tickIncrement(stop2, start2, count) : tickIncrement(start2, stop2, count);
  return (reverse ? -1 : 1) * (inc < 0 ? 1 / -inc : inc);
}
function range(start2, stop2, step) {
  start2 = +start2, stop2 = +stop2, step = (n = arguments.length) < 2 ? (stop2 = start2, start2 = 0, 1) : n < 3 ? 1 : +step;
  var i2 = -1, n = Math.max(0, Math.ceil((stop2 - start2) / step)) | 0, range2 = new Array(n);
  while (++i2 < n) {
    range2[i2] = start2 + i2 * step;
  }
  return range2;
}
function identity$3(x2) {
  return x2;
}
var top = 1, right = 2, bottom = 3, left = 4, epsilon$1 = 1e-6;
function translateX(x2) {
  return "translate(" + x2 + ",0)";
}
function translateY(y2) {
  return "translate(0," + y2 + ")";
}
function number$1(scale) {
  return (d) => +scale(d);
}
function center(scale, offset2) {
  offset2 = Math.max(0, scale.bandwidth() - offset2 * 2) / 2;
  if (scale.round())
    offset2 = Math.round(offset2);
  return (d) => +scale(d) + offset2;
}
function entering() {
  return !this.__axis;
}
function axis(orient, scale) {
  var tickArguments = [], tickValues = null, tickFormat2 = null, tickSizeInner = 6, tickSizeOuter = 6, tickPadding = 3, offset2 = typeof window !== "undefined" && window.devicePixelRatio > 1 ? 0 : 0.5, k = orient === top || orient === left ? -1 : 1, x2 = orient === left || orient === right ? "x" : "y", transform = orient === top || orient === bottom ? translateX : translateY;
  function axis2(context) {
    var values = tickValues == null ? scale.ticks ? scale.ticks.apply(scale, tickArguments) : scale.domain() : tickValues, format2 = tickFormat2 == null ? scale.tickFormat ? scale.tickFormat.apply(scale, tickArguments) : identity$3 : tickFormat2, spacing = Math.max(tickSizeInner, 0) + tickPadding, range2 = scale.range(), range0 = +range2[0] + offset2, range1 = +range2[range2.length - 1] + offset2, position = (scale.bandwidth ? center : number$1)(scale.copy(), offset2), selection2 = context.selection ? context.selection() : context, path = selection2.selectAll(".domain").data([null]), tick2 = selection2.selectAll(".tick").data(values, scale).order(), tickExit = tick2.exit(), tickEnter = tick2.enter().append("g").attr("class", "tick"), line2 = tick2.select("line"), text2 = tick2.select("text");
    path = path.merge(path.enter().insert("path", ".tick").attr("class", "domain").attr("stroke", "currentColor"));
    tick2 = tick2.merge(tickEnter);
    line2 = line2.merge(tickEnter.append("line").attr("stroke", "currentColor").attr(x2 + "2", k * tickSizeInner));
    text2 = text2.merge(tickEnter.append("text").attr("fill", "currentColor").attr(x2, k * spacing).attr("dy", orient === top ? "0em" : orient === bottom ? "0.71em" : "0.32em"));
    if (context !== selection2) {
      path = path.transition(context);
      tick2 = tick2.transition(context);
      line2 = line2.transition(context);
      text2 = text2.transition(context);
      tickExit = tickExit.transition(context).attr("opacity", epsilon$1).attr("transform", function(d) {
        return isFinite(d = position(d)) ? transform(d + offset2) : this.getAttribute("transform");
      });
      tickEnter.attr("opacity", epsilon$1).attr("transform", function(d) {
        var p = this.parentNode.__axis;
        return transform((p && isFinite(p = p(d)) ? p : position(d)) + offset2);
      });
    }
    tickExit.remove();
    path.attr("d", orient === left || orient === right ? tickSizeOuter ? "M" + k * tickSizeOuter + "," + range0 + "H" + offset2 + "V" + range1 + "H" + k * tickSizeOuter : "M" + offset2 + "," + range0 + "V" + range1 : tickSizeOuter ? "M" + range0 + "," + k * tickSizeOuter + "V" + offset2 + "H" + range1 + "V" + k * tickSizeOuter : "M" + range0 + "," + offset2 + "H" + range1);
    tick2.attr("opacity", 1).attr("transform", function(d) {
      return transform(position(d) + offset2);
    });
    line2.attr(x2 + "2", k * tickSizeInner);
    text2.attr(x2, k * spacing).text(format2);
    selection2.filter(entering).attr("fill", "none").attr("font-size", 10).attr("font-family", "sans-serif").attr("text-anchor", orient === right ? "start" : orient === left ? "end" : "middle");
    selection2.each(function() {
      this.__axis = position;
    });
  }
  axis2.scale = function(_) {
    return arguments.length ? (scale = _, axis2) : scale;
  };
  axis2.ticks = function() {
    return tickArguments = Array.from(arguments), axis2;
  };
  axis2.tickArguments = function(_) {
    return arguments.length ? (tickArguments = _ == null ? [] : Array.from(_), axis2) : tickArguments.slice();
  };
  axis2.tickValues = function(_) {
    return arguments.length ? (tickValues = _ == null ? null : Array.from(_), axis2) : tickValues && tickValues.slice();
  };
  axis2.tickFormat = function(_) {
    return arguments.length ? (tickFormat2 = _, axis2) : tickFormat2;
  };
  axis2.tickSize = function(_) {
    return arguments.length ? (tickSizeInner = tickSizeOuter = +_, axis2) : tickSizeInner;
  };
  axis2.tickSizeInner = function(_) {
    return arguments.length ? (tickSizeInner = +_, axis2) : tickSizeInner;
  };
  axis2.tickSizeOuter = function(_) {
    return arguments.length ? (tickSizeOuter = +_, axis2) : tickSizeOuter;
  };
  axis2.tickPadding = function(_) {
    return arguments.length ? (tickPadding = +_, axis2) : tickPadding;
  };
  axis2.offset = function(_) {
    return arguments.length ? (offset2 = +_, axis2) : offset2;
  };
  return axis2;
}
function axisBottom(scale) {
  return axis(bottom, scale);
}
function axisLeft(scale) {
  return axis(left, scale);
}
var noop$1 = { value: () => {
} };
function dispatch() {
  for (var i2 = 0, n = arguments.length, _ = {}, t; i2 < n; ++i2) {
    if (!(t = arguments[i2] + "") || t in _ || /[\s.]/.test(t))
      throw new Error("illegal type: " + t);
    _[t] = [];
  }
  return new Dispatch(_);
}
function Dispatch(_) {
  this._ = _;
}
function parseTypenames$1(typenames, types) {
  return typenames.trim().split(/^|\s+/).map(function(t) {
    var name2 = "", i2 = t.indexOf(".");
    if (i2 >= 0)
      name2 = t.slice(i2 + 1), t = t.slice(0, i2);
    if (t && !types.hasOwnProperty(t))
      throw new Error("unknown type: " + t);
    return { type: t, name: name2 };
  });
}
Dispatch.prototype = dispatch.prototype = {
  constructor: Dispatch,
  on: function(typename, callback) {
    var _ = this._, T = parseTypenames$1(typename + "", _), t, i2 = -1, n = T.length;
    if (arguments.length < 2) {
      while (++i2 < n)
        if ((t = (typename = T[i2]).type) && (t = get$1(_[t], typename.name)))
          return t;
      return;
    }
    if (callback != null && typeof callback !== "function")
      throw new Error("invalid callback: " + callback);
    while (++i2 < n) {
      if (t = (typename = T[i2]).type)
        _[t] = set$1(_[t], typename.name, callback);
      else if (callback == null)
        for (t in _)
          _[t] = set$1(_[t], typename.name, null);
    }
    return this;
  },
  copy: function() {
    var copy2 = {}, _ = this._;
    for (var t in _)
      copy2[t] = _[t].slice();
    return new Dispatch(copy2);
  },
  call: function(type, that) {
    if ((n = arguments.length - 2) > 0)
      for (var args = new Array(n), i2 = 0, n, t; i2 < n; ++i2)
        args[i2] = arguments[i2 + 2];
    if (!this._.hasOwnProperty(type))
      throw new Error("unknown type: " + type);
    for (t = this._[type], i2 = 0, n = t.length; i2 < n; ++i2)
      t[i2].value.apply(that, args);
  },
  apply: function(type, that, args) {
    if (!this._.hasOwnProperty(type))
      throw new Error("unknown type: " + type);
    for (var t = this._[type], i2 = 0, n = t.length; i2 < n; ++i2)
      t[i2].value.apply(that, args);
  }
};
function get$1(type, name2) {
  for (var i2 = 0, n = type.length, c; i2 < n; ++i2) {
    if ((c = type[i2]).name === name2) {
      return c.value;
    }
  }
}
function set$1(type, name2, callback) {
  for (var i2 = 0, n = type.length; i2 < n; ++i2) {
    if (type[i2].name === name2) {
      type[i2] = noop$1, type = type.slice(0, i2).concat(type.slice(i2 + 1));
      break;
    }
  }
  if (callback != null)
    type.push({ name: name2, value: callback });
  return type;
}
var xhtml = "http://www.w3.org/1999/xhtml";
const namespaces = {
  svg: "http://www.w3.org/2000/svg",
  xhtml,
  xlink: "http://www.w3.org/1999/xlink",
  xml: "http://www.w3.org/XML/1998/namespace",
  xmlns: "http://www.w3.org/2000/xmlns/"
};
function namespace(name2) {
  var prefix = name2 += "", i2 = prefix.indexOf(":");
  if (i2 >= 0 && (prefix = name2.slice(0, i2)) !== "xmlns")
    name2 = name2.slice(i2 + 1);
  return namespaces.hasOwnProperty(prefix) ? { space: namespaces[prefix], local: name2 } : name2;
}
function creatorInherit(name2) {
  return function() {
    var document2 = this.ownerDocument, uri = this.namespaceURI;
    return uri === xhtml && document2.documentElement.namespaceURI === xhtml ? document2.createElement(name2) : document2.createElementNS(uri, name2);
  };
}
function creatorFixed(fullname) {
  return function() {
    return this.ownerDocument.createElementNS(fullname.space, fullname.local);
  };
}
function creator(name2) {
  var fullname = namespace(name2);
  return (fullname.local ? creatorFixed : creatorInherit)(fullname);
}
function none() {
}
function selector(selector2) {
  return selector2 == null ? none : function() {
    return this.querySelector(selector2);
  };
}
function selection_select(select2) {
  if (typeof select2 !== "function")
    select2 = selector(select2);
  for (var groups = this._groups, m = groups.length, subgroups = new Array(m), j = 0; j < m; ++j) {
    for (var group2 = groups[j], n = group2.length, subgroup = subgroups[j] = new Array(n), node, subnode, i2 = 0; i2 < n; ++i2) {
      if ((node = group2[i2]) && (subnode = select2.call(node, node.__data__, i2, group2))) {
        if ("__data__" in node)
          subnode.__data__ = node.__data__;
        subgroup[i2] = subnode;
      }
    }
  }
  return new Selection$1(subgroups, this._parents);
}
function array$1(x2) {
  return x2 == null ? [] : Array.isArray(x2) ? x2 : Array.from(x2);
}
function empty() {
  return [];
}
function selectorAll(selector2) {
  return selector2 == null ? empty : function() {
    return this.querySelectorAll(selector2);
  };
}
function arrayAll(select2) {
  return function() {
    return array$1(select2.apply(this, arguments));
  };
}
function selection_selectAll(select2) {
  if (typeof select2 === "function")
    select2 = arrayAll(select2);
  else
    select2 = selectorAll(select2);
  for (var groups = this._groups, m = groups.length, subgroups = [], parents = [], j = 0; j < m; ++j) {
    for (var group2 = groups[j], n = group2.length, node, i2 = 0; i2 < n; ++i2) {
      if (node = group2[i2]) {
        subgroups.push(select2.call(node, node.__data__, i2, group2));
        parents.push(node);
      }
    }
  }
  return new Selection$1(subgroups, parents);
}
function matcher(selector2) {
  return function() {
    return this.matches(selector2);
  };
}
function childMatcher(selector2) {
  return function(node) {
    return node.matches(selector2);
  };
}
var find$1 = Array.prototype.find;
function childFind(match) {
  return function() {
    return find$1.call(this.children, match);
  };
}
function childFirst() {
  return this.firstElementChild;
}
function selection_selectChild(match) {
  return this.select(match == null ? childFirst : childFind(typeof match === "function" ? match : childMatcher(match)));
}
var filter = Array.prototype.filter;
function children() {
  return Array.from(this.children);
}
function childrenFilter(match) {
  return function() {
    return filter.call(this.children, match);
  };
}
function selection_selectChildren(match) {
  return this.selectAll(match == null ? children : childrenFilter(typeof match === "function" ? match : childMatcher(match)));
}
function selection_filter(match) {
  if (typeof match !== "function")
    match = matcher(match);
  for (var groups = this._groups, m = groups.length, subgroups = new Array(m), j = 0; j < m; ++j) {
    for (var group2 = groups[j], n = group2.length, subgroup = subgroups[j] = [], node, i2 = 0; i2 < n; ++i2) {
      if ((node = group2[i2]) && match.call(node, node.__data__, i2, group2)) {
        subgroup.push(node);
      }
    }
  }
  return new Selection$1(subgroups, this._parents);
}
function sparse(update2) {
  return new Array(update2.length);
}
function selection_enter() {
  return new Selection$1(this._enter || this._groups.map(sparse), this._parents);
}
function EnterNode(parent, datum2) {
  this.ownerDocument = parent.ownerDocument;
  this.namespaceURI = parent.namespaceURI;
  this._next = null;
  this._parent = parent;
  this.__data__ = datum2;
}
EnterNode.prototype = {
  constructor: EnterNode,
  appendChild: function(child) {
    return this._parent.insertBefore(child, this._next);
  },
  insertBefore: function(child, next2) {
    return this._parent.insertBefore(child, next2);
  },
  querySelector: function(selector2) {
    return this._parent.querySelector(selector2);
  },
  querySelectorAll: function(selector2) {
    return this._parent.querySelectorAll(selector2);
  }
};
function constant$2(x2) {
  return function() {
    return x2;
  };
}
function bindIndex(parent, group2, enter, update2, exit, data) {
  var i2 = 0, node, groupLength = group2.length, dataLength = data.length;
  for (; i2 < dataLength; ++i2) {
    if (node = group2[i2]) {
      node.__data__ = data[i2];
      update2[i2] = node;
    } else {
      enter[i2] = new EnterNode(parent, data[i2]);
    }
  }
  for (; i2 < groupLength; ++i2) {
    if (node = group2[i2]) {
      exit[i2] = node;
    }
  }
}
function bindKey(parent, group2, enter, update2, exit, data, key2) {
  var i2, node, nodeByKeyValue = /* @__PURE__ */ new Map(), groupLength = group2.length, dataLength = data.length, keyValues = new Array(groupLength), keyValue;
  for (i2 = 0; i2 < groupLength; ++i2) {
    if (node = group2[i2]) {
      keyValues[i2] = keyValue = key2.call(node, node.__data__, i2, group2) + "";
      if (nodeByKeyValue.has(keyValue)) {
        exit[i2] = node;
      } else {
        nodeByKeyValue.set(keyValue, node);
      }
    }
  }
  for (i2 = 0; i2 < dataLength; ++i2) {
    keyValue = key2.call(parent, data[i2], i2, data) + "";
    if (node = nodeByKeyValue.get(keyValue)) {
      update2[i2] = node;
      node.__data__ = data[i2];
      nodeByKeyValue.delete(keyValue);
    } else {
      enter[i2] = new EnterNode(parent, data[i2]);
    }
  }
  for (i2 = 0; i2 < groupLength; ++i2) {
    if ((node = group2[i2]) && nodeByKeyValue.get(keyValues[i2]) === node) {
      exit[i2] = node;
    }
  }
}
function datum(node) {
  return node.__data__;
}
function selection_data(value, key2) {
  if (!arguments.length)
    return Array.from(this, datum);
  var bind2 = key2 ? bindKey : bindIndex, parents = this._parents, groups = this._groups;
  if (typeof value !== "function")
    value = constant$2(value);
  for (var m = groups.length, update2 = new Array(m), enter = new Array(m), exit = new Array(m), j = 0; j < m; ++j) {
    var parent = parents[j], group2 = groups[j], groupLength = group2.length, data = arraylike(value.call(parent, parent && parent.__data__, j, parents)), dataLength = data.length, enterGroup = enter[j] = new Array(dataLength), updateGroup = update2[j] = new Array(dataLength), exitGroup = exit[j] = new Array(groupLength);
    bind2(parent, group2, enterGroup, updateGroup, exitGroup, data, key2);
    for (var i0 = 0, i1 = 0, previous, next2; i0 < dataLength; ++i0) {
      if (previous = enterGroup[i0]) {
        if (i0 >= i1)
          i1 = i0 + 1;
        while (!(next2 = updateGroup[i1]) && ++i1 < dataLength)
          ;
        previous._next = next2 || null;
      }
    }
  }
  update2 = new Selection$1(update2, parents);
  update2._enter = enter;
  update2._exit = exit;
  return update2;
}
function arraylike(data) {
  return typeof data === "object" && "length" in data ? data : Array.from(data);
}
function selection_exit() {
  return new Selection$1(this._exit || this._groups.map(sparse), this._parents);
}
function selection_join(onenter, onupdate, onexit) {
  var enter = this.enter(), update2 = this, exit = this.exit();
  if (typeof onenter === "function") {
    enter = onenter(enter);
    if (enter)
      enter = enter.selection();
  } else {
    enter = enter.append(onenter + "");
  }
  if (onupdate != null) {
    update2 = onupdate(update2);
    if (update2)
      update2 = update2.selection();
  }
  if (onexit == null)
    exit.remove();
  else
    onexit(exit);
  return enter && update2 ? enter.merge(update2).order() : update2;
}
function selection_merge(context) {
  var selection2 = context.selection ? context.selection() : context;
  for (var groups0 = this._groups, groups1 = selection2._groups, m0 = groups0.length, m1 = groups1.length, m = Math.min(m0, m1), merges = new Array(m0), j = 0; j < m; ++j) {
    for (var group0 = groups0[j], group1 = groups1[j], n = group0.length, merge = merges[j] = new Array(n), node, i2 = 0; i2 < n; ++i2) {
      if (node = group0[i2] || group1[i2]) {
        merge[i2] = node;
      }
    }
  }
  for (; j < m0; ++j) {
    merges[j] = groups0[j];
  }
  return new Selection$1(merges, this._parents);
}
function selection_order() {
  for (var groups = this._groups, j = -1, m = groups.length; ++j < m; ) {
    for (var group2 = groups[j], i2 = group2.length - 1, next2 = group2[i2], node; --i2 >= 0; ) {
      if (node = group2[i2]) {
        if (next2 && node.compareDocumentPosition(next2) ^ 4)
          next2.parentNode.insertBefore(node, next2);
        next2 = node;
      }
    }
  }
  return this;
}
function selection_sort(compare2) {
  if (!compare2)
    compare2 = ascending;
  function compareNode(a, b) {
    return a && b ? compare2(a.__data__, b.__data__) : !a - !b;
  }
  for (var groups = this._groups, m = groups.length, sortgroups = new Array(m), j = 0; j < m; ++j) {
    for (var group2 = groups[j], n = group2.length, sortgroup = sortgroups[j] = new Array(n), node, i2 = 0; i2 < n; ++i2) {
      if (node = group2[i2]) {
        sortgroup[i2] = node;
      }
    }
    sortgroup.sort(compareNode);
  }
  return new Selection$1(sortgroups, this._parents).order();
}
function ascending(a, b) {
  return a < b ? -1 : a > b ? 1 : a >= b ? 0 : NaN;
}
function selection_call() {
  var callback = arguments[0];
  arguments[0] = this;
  callback.apply(null, arguments);
  return this;
}
function selection_nodes() {
  return Array.from(this);
}
function selection_node() {
  for (var groups = this._groups, j = 0, m = groups.length; j < m; ++j) {
    for (var group2 = groups[j], i2 = 0, n = group2.length; i2 < n; ++i2) {
      var node = group2[i2];
      if (node)
        return node;
    }
  }
  return null;
}
function selection_size() {
  let size2 = 0;
  for (const node of this)
    ++size2;
  return size2;
}
function selection_empty() {
  return !this.node();
}
function selection_each(callback) {
  for (var groups = this._groups, j = 0, m = groups.length; j < m; ++j) {
    for (var group2 = groups[j], i2 = 0, n = group2.length, node; i2 < n; ++i2) {
      if (node = group2[i2])
        callback.call(node, node.__data__, i2, group2);
    }
  }
  return this;
}
function attrRemove$1(name2) {
  return function() {
    this.removeAttribute(name2);
  };
}
function attrRemoveNS$1(fullname) {
  return function() {
    this.removeAttributeNS(fullname.space, fullname.local);
  };
}
function attrConstant$1(name2, value) {
  return function() {
    this.setAttribute(name2, value);
  };
}
function attrConstantNS$1(fullname, value) {
  return function() {
    this.setAttributeNS(fullname.space, fullname.local, value);
  };
}
function attrFunction$1(name2, value) {
  return function() {
    var v = value.apply(this, arguments);
    if (v == null)
      this.removeAttribute(name2);
    else
      this.setAttribute(name2, v);
  };
}
function attrFunctionNS$1(fullname, value) {
  return function() {
    var v = value.apply(this, arguments);
    if (v == null)
      this.removeAttributeNS(fullname.space, fullname.local);
    else
      this.setAttributeNS(fullname.space, fullname.local, v);
  };
}
function selection_attr(name2, value) {
  var fullname = namespace(name2);
  if (arguments.length < 2) {
    var node = this.node();
    return fullname.local ? node.getAttributeNS(fullname.space, fullname.local) : node.getAttribute(fullname);
  }
  return this.each((value == null ? fullname.local ? attrRemoveNS$1 : attrRemove$1 : typeof value === "function" ? fullname.local ? attrFunctionNS$1 : attrFunction$1 : fullname.local ? attrConstantNS$1 : attrConstant$1)(fullname, value));
}
function defaultView(node) {
  return node.ownerDocument && node.ownerDocument.defaultView || node.document && node || node.defaultView;
}
function styleRemove$1(name2) {
  return function() {
    this.style.removeProperty(name2);
  };
}
function styleConstant$1(name2, value, priority) {
  return function() {
    this.style.setProperty(name2, value, priority);
  };
}
function styleFunction$1(name2, value, priority) {
  return function() {
    var v = value.apply(this, arguments);
    if (v == null)
      this.style.removeProperty(name2);
    else
      this.style.setProperty(name2, v, priority);
  };
}
function selection_style(name2, value, priority) {
  return arguments.length > 1 ? this.each((value == null ? styleRemove$1 : typeof value === "function" ? styleFunction$1 : styleConstant$1)(name2, value, priority == null ? "" : priority)) : styleValue(this.node(), name2);
}
function styleValue(node, name2) {
  return node.style.getPropertyValue(name2) || defaultView(node).getComputedStyle(node, null).getPropertyValue(name2);
}
function propertyRemove(name2) {
  return function() {
    delete this[name2];
  };
}
function propertyConstant(name2, value) {
  return function() {
    this[name2] = value;
  };
}
function propertyFunction(name2, value) {
  return function() {
    var v = value.apply(this, arguments);
    if (v == null)
      delete this[name2];
    else
      this[name2] = v;
  };
}
function selection_property(name2, value) {
  return arguments.length > 1 ? this.each((value == null ? propertyRemove : typeof value === "function" ? propertyFunction : propertyConstant)(name2, value)) : this.node()[name2];
}
function classArray(string) {
  return string.trim().split(/^|\s+/);
}
function classList(node) {
  return node.classList || new ClassList(node);
}
function ClassList(node) {
  this._node = node;
  this._names = classArray(node.getAttribute("class") || "");
}
ClassList.prototype = {
  add: function(name2) {
    var i2 = this._names.indexOf(name2);
    if (i2 < 0) {
      this._names.push(name2);
      this._node.setAttribute("class", this._names.join(" "));
    }
  },
  remove: function(name2) {
    var i2 = this._names.indexOf(name2);
    if (i2 >= 0) {
      this._names.splice(i2, 1);
      this._node.setAttribute("class", this._names.join(" "));
    }
  },
  contains: function(name2) {
    return this._names.indexOf(name2) >= 0;
  }
};
function classedAdd(node, names) {
  var list2 = classList(node), i2 = -1, n = names.length;
  while (++i2 < n)
    list2.add(names[i2]);
}
function classedRemove(node, names) {
  var list2 = classList(node), i2 = -1, n = names.length;
  while (++i2 < n)
    list2.remove(names[i2]);
}
function classedTrue(names) {
  return function() {
    classedAdd(this, names);
  };
}
function classedFalse(names) {
  return function() {
    classedRemove(this, names);
  };
}
function classedFunction(names, value) {
  return function() {
    (value.apply(this, arguments) ? classedAdd : classedRemove)(this, names);
  };
}
function selection_classed(name2, value) {
  var names = classArray(name2 + "");
  if (arguments.length < 2) {
    var list2 = classList(this.node()), i2 = -1, n = names.length;
    while (++i2 < n)
      if (!list2.contains(names[i2]))
        return false;
    return true;
  }
  return this.each((typeof value === "function" ? classedFunction : value ? classedTrue : classedFalse)(names, value));
}
function textRemove() {
  this.textContent = "";
}
function textConstant$1(value) {
  return function() {
    this.textContent = value;
  };
}
function textFunction$1(value) {
  return function() {
    var v = value.apply(this, arguments);
    this.textContent = v == null ? "" : v;
  };
}
function selection_text(value) {
  return arguments.length ? this.each(value == null ? textRemove : (typeof value === "function" ? textFunction$1 : textConstant$1)(value)) : this.node().textContent;
}
function htmlRemove() {
  this.innerHTML = "";
}
function htmlConstant(value) {
  return function() {
    this.innerHTML = value;
  };
}
function htmlFunction(value) {
  return function() {
    var v = value.apply(this, arguments);
    this.innerHTML = v == null ? "" : v;
  };
}
function selection_html(value) {
  return arguments.length ? this.each(value == null ? htmlRemove : (typeof value === "function" ? htmlFunction : htmlConstant)(value)) : this.node().innerHTML;
}
function raise() {
  if (this.nextSibling)
    this.parentNode.appendChild(this);
}
function selection_raise() {
  return this.each(raise);
}
function lower() {
  if (this.previousSibling)
    this.parentNode.insertBefore(this, this.parentNode.firstChild);
}
function selection_lower() {
  return this.each(lower);
}
function selection_append(name2) {
  var create2 = typeof name2 === "function" ? name2 : creator(name2);
  return this.select(function() {
    return this.appendChild(create2.apply(this, arguments));
  });
}
function constantNull() {
  return null;
}
function selection_insert(name2, before) {
  var create2 = typeof name2 === "function" ? name2 : creator(name2), select2 = before == null ? constantNull : typeof before === "function" ? before : selector(before);
  return this.select(function() {
    return this.insertBefore(create2.apply(this, arguments), select2.apply(this, arguments) || null);
  });
}
function remove$1() {
  var parent = this.parentNode;
  if (parent)
    parent.removeChild(this);
}
function selection_remove() {
  return this.each(remove$1);
}
function selection_cloneShallow() {
  var clone = this.cloneNode(false), parent = this.parentNode;
  return parent ? parent.insertBefore(clone, this.nextSibling) : clone;
}
function selection_cloneDeep() {
  var clone = this.cloneNode(true), parent = this.parentNode;
  return parent ? parent.insertBefore(clone, this.nextSibling) : clone;
}
function selection_clone(deep) {
  return this.select(deep ? selection_cloneDeep : selection_cloneShallow);
}
function selection_datum(value) {
  return arguments.length ? this.property("__data__", value) : this.node().__data__;
}
function contextListener(listener) {
  return function(event) {
    listener.call(this, event, this.__data__);
  };
}
function parseTypenames(typenames) {
  return typenames.trim().split(/^|\s+/).map(function(t) {
    var name2 = "", i2 = t.indexOf(".");
    if (i2 >= 0)
      name2 = t.slice(i2 + 1), t = t.slice(0, i2);
    return { type: t, name: name2 };
  });
}
function onRemove(typename) {
  return function() {
    var on = this.__on;
    if (!on)
      return;
    for (var j = 0, i2 = -1, m = on.length, o; j < m; ++j) {
      if (o = on[j], (!typename.type || o.type === typename.type) && o.name === typename.name) {
        this.removeEventListener(o.type, o.listener, o.options);
      } else {
        on[++i2] = o;
      }
    }
    if (++i2)
      on.length = i2;
    else
      delete this.__on;
  };
}
function onAdd(typename, value, options) {
  return function() {
    var on = this.__on, o, listener = contextListener(value);
    if (on)
      for (var j = 0, m = on.length; j < m; ++j) {
        if ((o = on[j]).type === typename.type && o.name === typename.name) {
          this.removeEventListener(o.type, o.listener, o.options);
          this.addEventListener(o.type, o.listener = listener, o.options = options);
          o.value = value;
          return;
        }
      }
    this.addEventListener(typename.type, listener, options);
    o = { type: typename.type, name: typename.name, value, listener, options };
    if (!on)
      this.__on = [o];
    else
      on.push(o);
  };
}
function selection_on(typename, value, options) {
  var typenames = parseTypenames(typename + ""), i2, n = typenames.length, t;
  if (arguments.length < 2) {
    var on = this.node().__on;
    if (on)
      for (var j = 0, m = on.length, o; j < m; ++j) {
        for (i2 = 0, o = on[j]; i2 < n; ++i2) {
          if ((t = typenames[i2]).type === o.type && t.name === o.name) {
            return o.value;
          }
        }
      }
    return;
  }
  on = value ? onAdd : onRemove;
  for (i2 = 0; i2 < n; ++i2)
    this.each(on(typenames[i2], value, options));
  return this;
}
function dispatchEvent(node, type, params) {
  var window2 = defaultView(node), event = window2.CustomEvent;
  if (typeof event === "function") {
    event = new event(type, params);
  } else {
    event = window2.document.createEvent("Event");
    if (params)
      event.initEvent(type, params.bubbles, params.cancelable), event.detail = params.detail;
    else
      event.initEvent(type, false, false);
  }
  node.dispatchEvent(event);
}
function dispatchConstant(type, params) {
  return function() {
    return dispatchEvent(this, type, params);
  };
}
function dispatchFunction(type, params) {
  return function() {
    return dispatchEvent(this, type, params.apply(this, arguments));
  };
}
function selection_dispatch(type, params) {
  return this.each((typeof params === "function" ? dispatchFunction : dispatchConstant)(type, params));
}
function* selection_iterator() {
  for (var groups = this._groups, j = 0, m = groups.length; j < m; ++j) {
    for (var group2 = groups[j], i2 = 0, n = group2.length, node; i2 < n; ++i2) {
      if (node = group2[i2])
        yield node;
    }
  }
}
var root = [null];
function Selection$1(groups, parents) {
  this._groups = groups;
  this._parents = parents;
}
function selection() {
  return new Selection$1([[document.documentElement]], root);
}
function selection_selection() {
  return this;
}
Selection$1.prototype = selection.prototype = {
  constructor: Selection$1,
  select: selection_select,
  selectAll: selection_selectAll,
  selectChild: selection_selectChild,
  selectChildren: selection_selectChildren,
  filter: selection_filter,
  data: selection_data,
  enter: selection_enter,
  exit: selection_exit,
  join: selection_join,
  merge: selection_merge,
  selection: selection_selection,
  order: selection_order,
  sort: selection_sort,
  call: selection_call,
  nodes: selection_nodes,
  node: selection_node,
  size: selection_size,
  empty: selection_empty,
  each: selection_each,
  attr: selection_attr,
  style: selection_style,
  property: selection_property,
  classed: selection_classed,
  text: selection_text,
  html: selection_html,
  raise: selection_raise,
  lower: selection_lower,
  append: selection_append,
  insert: selection_insert,
  remove: selection_remove,
  clone: selection_clone,
  datum: selection_datum,
  on: selection_on,
  dispatch: selection_dispatch,
  [Symbol.iterator]: selection_iterator
};
function select(selector2) {
  return typeof selector2 === "string" ? new Selection$1([[document.querySelector(selector2)]], [document.documentElement]) : new Selection$1([[selector2]], root);
}
function define(constructor, factory, prototype) {
  constructor.prototype = factory.prototype = prototype;
  prototype.constructor = constructor;
}
function extend(parent, definition) {
  var prototype = Object.create(parent.prototype);
  for (var key2 in definition)
    prototype[key2] = definition[key2];
  return prototype;
}
function Color() {
}
var darker = 0.7;
var brighter = 1 / darker;
var reI = "\\s*([+-]?\\d+)\\s*", reN = "\\s*([+-]?(?:\\d*\\.)?\\d+(?:[eE][+-]?\\d+)?)\\s*", reP = "\\s*([+-]?(?:\\d*\\.)?\\d+(?:[eE][+-]?\\d+)?)%\\s*", reHex = /^#([0-9a-f]{3,8})$/, reRgbInteger = new RegExp(`^rgb\\(${reI},${reI},${reI}\\)$`), reRgbPercent = new RegExp(`^rgb\\(${reP},${reP},${reP}\\)$`), reRgbaInteger = new RegExp(`^rgba\\(${reI},${reI},${reI},${reN}\\)$`), reRgbaPercent = new RegExp(`^rgba\\(${reP},${reP},${reP},${reN}\\)$`), reHslPercent = new RegExp(`^hsl\\(${reN},${reP},${reP}\\)$`), reHslaPercent = new RegExp(`^hsla\\(${reN},${reP},${reP},${reN}\\)$`);
var named = {
  aliceblue: 15792383,
  antiquewhite: 16444375,
  aqua: 65535,
  aquamarine: 8388564,
  azure: 15794175,
  beige: 16119260,
  bisque: 16770244,
  black: 0,
  blanchedalmond: 16772045,
  blue: 255,
  blueviolet: 9055202,
  brown: 10824234,
  burlywood: 14596231,
  cadetblue: 6266528,
  chartreuse: 8388352,
  chocolate: 13789470,
  coral: 16744272,
  cornflowerblue: 6591981,
  cornsilk: 16775388,
  crimson: 14423100,
  cyan: 65535,
  darkblue: 139,
  darkcyan: 35723,
  darkgoldenrod: 12092939,
  darkgray: 11119017,
  darkgreen: 25600,
  darkgrey: 11119017,
  darkkhaki: 12433259,
  darkmagenta: 9109643,
  darkolivegreen: 5597999,
  darkorange: 16747520,
  darkorchid: 10040012,
  darkred: 9109504,
  darksalmon: 15308410,
  darkseagreen: 9419919,
  darkslateblue: 4734347,
  darkslategray: 3100495,
  darkslategrey: 3100495,
  darkturquoise: 52945,
  darkviolet: 9699539,
  deeppink: 16716947,
  deepskyblue: 49151,
  dimgray: 6908265,
  dimgrey: 6908265,
  dodgerblue: 2003199,
  firebrick: 11674146,
  floralwhite: 16775920,
  forestgreen: 2263842,
  fuchsia: 16711935,
  gainsboro: 14474460,
  ghostwhite: 16316671,
  gold: 16766720,
  goldenrod: 14329120,
  gray: 8421504,
  green: 32768,
  greenyellow: 11403055,
  grey: 8421504,
  honeydew: 15794160,
  hotpink: 16738740,
  indianred: 13458524,
  indigo: 4915330,
  ivory: 16777200,
  khaki: 15787660,
  lavender: 15132410,
  lavenderblush: 16773365,
  lawngreen: 8190976,
  lemonchiffon: 16775885,
  lightblue: 11393254,
  lightcoral: 15761536,
  lightcyan: 14745599,
  lightgoldenrodyellow: 16448210,
  lightgray: 13882323,
  lightgreen: 9498256,
  lightgrey: 13882323,
  lightpink: 16758465,
  lightsalmon: 16752762,
  lightseagreen: 2142890,
  lightskyblue: 8900346,
  lightslategray: 7833753,
  lightslategrey: 7833753,
  lightsteelblue: 11584734,
  lightyellow: 16777184,
  lime: 65280,
  limegreen: 3329330,
  linen: 16445670,
  magenta: 16711935,
  maroon: 8388608,
  mediumaquamarine: 6737322,
  mediumblue: 205,
  mediumorchid: 12211667,
  mediumpurple: 9662683,
  mediumseagreen: 3978097,
  mediumslateblue: 8087790,
  mediumspringgreen: 64154,
  mediumturquoise: 4772300,
  mediumvioletred: 13047173,
  midnightblue: 1644912,
  mintcream: 16121850,
  mistyrose: 16770273,
  moccasin: 16770229,
  navajowhite: 16768685,
  navy: 128,
  oldlace: 16643558,
  olive: 8421376,
  olivedrab: 7048739,
  orange: 16753920,
  orangered: 16729344,
  orchid: 14315734,
  palegoldenrod: 15657130,
  palegreen: 10025880,
  paleturquoise: 11529966,
  palevioletred: 14381203,
  papayawhip: 16773077,
  peachpuff: 16767673,
  peru: 13468991,
  pink: 16761035,
  plum: 14524637,
  powderblue: 11591910,
  purple: 8388736,
  rebeccapurple: 6697881,
  red: 16711680,
  rosybrown: 12357519,
  royalblue: 4286945,
  saddlebrown: 9127187,
  salmon: 16416882,
  sandybrown: 16032864,
  seagreen: 3050327,
  seashell: 16774638,
  sienna: 10506797,
  silver: 12632256,
  skyblue: 8900331,
  slateblue: 6970061,
  slategray: 7372944,
  slategrey: 7372944,
  snow: 16775930,
  springgreen: 65407,
  steelblue: 4620980,
  tan: 13808780,
  teal: 32896,
  thistle: 14204888,
  tomato: 16737095,
  turquoise: 4251856,
  violet: 15631086,
  wheat: 16113331,
  white: 16777215,
  whitesmoke: 16119285,
  yellow: 16776960,
  yellowgreen: 10145074
};
define(Color, color, {
  copy(channels) {
    return Object.assign(new this.constructor(), this, channels);
  },
  displayable() {
    return this.rgb().displayable();
  },
  hex: color_formatHex,
  // Deprecated! Use color.formatHex.
  formatHex: color_formatHex,
  formatHex8: color_formatHex8,
  formatHsl: color_formatHsl,
  formatRgb: color_formatRgb,
  toString: color_formatRgb
});
function color_formatHex() {
  return this.rgb().formatHex();
}
function color_formatHex8() {
  return this.rgb().formatHex8();
}
function color_formatHsl() {
  return hslConvert(this).formatHsl();
}
function color_formatRgb() {
  return this.rgb().formatRgb();
}
function color(format2) {
  var m, l;
  format2 = (format2 + "").trim().toLowerCase();
  return (m = reHex.exec(format2)) ? (l = m[1].length, m = parseInt(m[1], 16), l === 6 ? rgbn(m) : l === 3 ? new Rgb(m >> 8 & 15 | m >> 4 & 240, m >> 4 & 15 | m & 240, (m & 15) << 4 | m & 15, 1) : l === 8 ? rgba(m >> 24 & 255, m >> 16 & 255, m >> 8 & 255, (m & 255) / 255) : l === 4 ? rgba(m >> 12 & 15 | m >> 8 & 240, m >> 8 & 15 | m >> 4 & 240, m >> 4 & 15 | m & 240, ((m & 15) << 4 | m & 15) / 255) : null) : (m = reRgbInteger.exec(format2)) ? new Rgb(m[1], m[2], m[3], 1) : (m = reRgbPercent.exec(format2)) ? new Rgb(m[1] * 255 / 100, m[2] * 255 / 100, m[3] * 255 / 100, 1) : (m = reRgbaInteger.exec(format2)) ? rgba(m[1], m[2], m[3], m[4]) : (m = reRgbaPercent.exec(format2)) ? rgba(m[1] * 255 / 100, m[2] * 255 / 100, m[3] * 255 / 100, m[4]) : (m = reHslPercent.exec(format2)) ? hsla(m[1], m[2] / 100, m[3] / 100, 1) : (m = reHslaPercent.exec(format2)) ? hsla(m[1], m[2] / 100, m[3] / 100, m[4]) : named.hasOwnProperty(format2) ? rgbn(named[format2]) : format2 === "transparent" ? new Rgb(NaN, NaN, NaN, 0) : null;
}
function rgbn(n) {
  return new Rgb(n >> 16 & 255, n >> 8 & 255, n & 255, 1);
}
function rgba(r, g, b, a) {
  if (a <= 0)
    r = g = b = NaN;
  return new Rgb(r, g, b, a);
}
function rgbConvert(o) {
  if (!(o instanceof Color))
    o = color(o);
  if (!o)
    return new Rgb();
  o = o.rgb();
  return new Rgb(o.r, o.g, o.b, o.opacity);
}
function rgb(r, g, b, opacity2) {
  return arguments.length === 1 ? rgbConvert(r) : new Rgb(r, g, b, opacity2 == null ? 1 : opacity2);
}
function Rgb(r, g, b, opacity2) {
  this.r = +r;
  this.g = +g;
  this.b = +b;
  this.opacity = +opacity2;
}
define(Rgb, rgb, extend(Color, {
  brighter(k) {
    k = k == null ? brighter : Math.pow(brighter, k);
    return new Rgb(this.r * k, this.g * k, this.b * k, this.opacity);
  },
  darker(k) {
    k = k == null ? darker : Math.pow(darker, k);
    return new Rgb(this.r * k, this.g * k, this.b * k, this.opacity);
  },
  rgb() {
    return this;
  },
  clamp() {
    return new Rgb(clampi(this.r), clampi(this.g), clampi(this.b), clampa(this.opacity));
  },
  displayable() {
    return -0.5 <= this.r && this.r < 255.5 && (-0.5 <= this.g && this.g < 255.5) && (-0.5 <= this.b && this.b < 255.5) && (0 <= this.opacity && this.opacity <= 1);
  },
  hex: rgb_formatHex,
  // Deprecated! Use color.formatHex.
  formatHex: rgb_formatHex,
  formatHex8: rgb_formatHex8,
  formatRgb: rgb_formatRgb,
  toString: rgb_formatRgb
}));
function rgb_formatHex() {
  return `#${hex(this.r)}${hex(this.g)}${hex(this.b)}`;
}
function rgb_formatHex8() {
  return `#${hex(this.r)}${hex(this.g)}${hex(this.b)}${hex((isNaN(this.opacity) ? 1 : this.opacity) * 255)}`;
}
function rgb_formatRgb() {
  const a = clampa(this.opacity);
  return `${a === 1 ? "rgb(" : "rgba("}${clampi(this.r)}, ${clampi(this.g)}, ${clampi(this.b)}${a === 1 ? ")" : `, ${a})`}`;
}
function clampa(opacity2) {
  return isNaN(opacity2) ? 1 : Math.max(0, Math.min(1, opacity2));
}
function clampi(value) {
  return Math.max(0, Math.min(255, Math.round(value) || 0));
}
function hex(value) {
  value = clampi(value);
  return (value < 16 ? "0" : "") + value.toString(16);
}
function hsla(h, s, l, a) {
  if (a <= 0)
    h = s = l = NaN;
  else if (l <= 0 || l >= 1)
    h = s = NaN;
  else if (s <= 0)
    h = NaN;
  return new Hsl(h, s, l, a);
}
function hslConvert(o) {
  if (o instanceof Hsl)
    return new Hsl(o.h, o.s, o.l, o.opacity);
  if (!(o instanceof Color))
    o = color(o);
  if (!o)
    return new Hsl();
  if (o instanceof Hsl)
    return o;
  o = o.rgb();
  var r = o.r / 255, g = o.g / 255, b = o.b / 255, min2 = Math.min(r, g, b), max2 = Math.max(r, g, b), h = NaN, s = max2 - min2, l = (max2 + min2) / 2;
  if (s) {
    if (r === max2)
      h = (g - b) / s + (g < b) * 6;
    else if (g === max2)
      h = (b - r) / s + 2;
    else
      h = (r - g) / s + 4;
    s /= l < 0.5 ? max2 + min2 : 2 - max2 - min2;
    h *= 60;
  } else {
    s = l > 0 && l < 1 ? 0 : h;
  }
  return new Hsl(h, s, l, o.opacity);
}
function hsl(h, s, l, opacity2) {
  return arguments.length === 1 ? hslConvert(h) : new Hsl(h, s, l, opacity2 == null ? 1 : opacity2);
}
function Hsl(h, s, l, opacity2) {
  this.h = +h;
  this.s = +s;
  this.l = +l;
  this.opacity = +opacity2;
}
define(Hsl, hsl, extend(Color, {
  brighter(k) {
    k = k == null ? brighter : Math.pow(brighter, k);
    return new Hsl(this.h, this.s, this.l * k, this.opacity);
  },
  darker(k) {
    k = k == null ? darker : Math.pow(darker, k);
    return new Hsl(this.h, this.s, this.l * k, this.opacity);
  },
  rgb() {
    var h = this.h % 360 + (this.h < 0) * 360, s = isNaN(h) || isNaN(this.s) ? 0 : this.s, l = this.l, m2 = l + (l < 0.5 ? l : 1 - l) * s, m1 = 2 * l - m2;
    return new Rgb(
      hsl2rgb(h >= 240 ? h - 240 : h + 120, m1, m2),
      hsl2rgb(h, m1, m2),
      hsl2rgb(h < 120 ? h + 240 : h - 120, m1, m2),
      this.opacity
    );
  },
  clamp() {
    return new Hsl(clamph(this.h), clampt(this.s), clampt(this.l), clampa(this.opacity));
  },
  displayable() {
    return (0 <= this.s && this.s <= 1 || isNaN(this.s)) && (0 <= this.l && this.l <= 1) && (0 <= this.opacity && this.opacity <= 1);
  },
  formatHsl() {
    const a = clampa(this.opacity);
    return `${a === 1 ? "hsl(" : "hsla("}${clamph(this.h)}, ${clampt(this.s) * 100}%, ${clampt(this.l) * 100}%${a === 1 ? ")" : `, ${a})`}`;
  }
}));
function clamph(value) {
  value = (value || 0) % 360;
  return value < 0 ? value + 360 : value;
}
function clampt(value) {
  return Math.max(0, Math.min(1, value || 0));
}
function hsl2rgb(h, m1, m2) {
  return (h < 60 ? m1 + (m2 - m1) * h / 60 : h < 180 ? m2 : h < 240 ? m1 + (m2 - m1) * (240 - h) / 60 : m1) * 255;
}
const constant$1 = (x2) => () => x2;
function linear$1(a, d) {
  return function(t) {
    return a + t * d;
  };
}
function exponential(a, b, y2) {
  return a = Math.pow(a, y2), b = Math.pow(b, y2) - a, y2 = 1 / y2, function(t) {
    return Math.pow(a + t * b, y2);
  };
}
function gamma(y2) {
  return (y2 = +y2) === 1 ? nogamma : function(a, b) {
    return b - a ? exponential(a, b, y2) : constant$1(isNaN(a) ? b : a);
  };
}
function nogamma(a, b) {
  var d = b - a;
  return d ? linear$1(a, d) : constant$1(isNaN(a) ? b : a);
}
const interpolateRgb = function rgbGamma(y2) {
  var color2 = gamma(y2);
  function rgb$1(start2, end) {
    var r = color2((start2 = rgb(start2)).r, (end = rgb(end)).r), g = color2(start2.g, end.g), b = color2(start2.b, end.b), opacity2 = nogamma(start2.opacity, end.opacity);
    return function(t) {
      start2.r = r(t);
      start2.g = g(t);
      start2.b = b(t);
      start2.opacity = opacity2(t);
      return start2 + "";
    };
  }
  rgb$1.gamma = rgbGamma;
  return rgb$1;
}(1);
function numberArray(a, b) {
  if (!b)
    b = [];
  var n = a ? Math.min(b.length, a.length) : 0, c = b.slice(), i2;
  return function(t) {
    for (i2 = 0; i2 < n; ++i2)
      c[i2] = a[i2] * (1 - t) + b[i2] * t;
    return c;
  };
}
function isNumberArray(x2) {
  return ArrayBuffer.isView(x2) && !(x2 instanceof DataView);
}
function genericArray(a, b) {
  var nb = b ? b.length : 0, na = a ? Math.min(nb, a.length) : 0, x2 = new Array(na), c = new Array(nb), i2;
  for (i2 = 0; i2 < na; ++i2)
    x2[i2] = interpolate$1(a[i2], b[i2]);
  for (; i2 < nb; ++i2)
    c[i2] = b[i2];
  return function(t) {
    for (i2 = 0; i2 < na; ++i2)
      c[i2] = x2[i2](t);
    return c;
  };
}
function date(a, b) {
  var d = /* @__PURE__ */ new Date();
  return a = +a, b = +b, function(t) {
    return d.setTime(a * (1 - t) + b * t), d;
  };
}
function interpolateNumber(a, b) {
  return a = +a, b = +b, function(t) {
    return a * (1 - t) + b * t;
  };
}
function object(a, b) {
  var i2 = {}, c = {}, k;
  if (a === null || typeof a !== "object")
    a = {};
  if (b === null || typeof b !== "object")
    b = {};
  for (k in b) {
    if (k in a) {
      i2[k] = interpolate$1(a[k], b[k]);
    } else {
      c[k] = b[k];
    }
  }
  return function(t) {
    for (k in i2)
      c[k] = i2[k](t);
    return c;
  };
}
var reA = /[-+]?(?:\d+\.?\d*|\.?\d+)(?:[eE][-+]?\d+)?/g, reB = new RegExp(reA.source, "g");
function zero(b) {
  return function() {
    return b;
  };
}
function one(b) {
  return function(t) {
    return b(t) + "";
  };
}
function interpolateString(a, b) {
  var bi = reA.lastIndex = reB.lastIndex = 0, am, bm, bs, i2 = -1, s = [], q = [];
  a = a + "", b = b + "";
  while ((am = reA.exec(a)) && (bm = reB.exec(b))) {
    if ((bs = bm.index) > bi) {
      bs = b.slice(bi, bs);
      if (s[i2])
        s[i2] += bs;
      else
        s[++i2] = bs;
    }
    if ((am = am[0]) === (bm = bm[0])) {
      if (s[i2])
        s[i2] += bm;
      else
        s[++i2] = bm;
    } else {
      s[++i2] = null;
      q.push({ i: i2, x: interpolateNumber(am, bm) });
    }
    bi = reB.lastIndex;
  }
  if (bi < b.length) {
    bs = b.slice(bi);
    if (s[i2])
      s[i2] += bs;
    else
      s[++i2] = bs;
  }
  return s.length < 2 ? q[0] ? one(q[0].x) : zero(b) : (b = q.length, function(t) {
    for (var i3 = 0, o; i3 < b; ++i3)
      s[(o = q[i3]).i] = o.x(t);
    return s.join("");
  });
}
function interpolate$1(a, b) {
  var t = typeof b, c;
  return b == null || t === "boolean" ? constant$1(b) : (t === "number" ? interpolateNumber : t === "string" ? (c = color(b)) ? (b = c, interpolateRgb) : interpolateString : b instanceof color ? interpolateRgb : b instanceof Date ? date : isNumberArray(b) ? numberArray : Array.isArray(b) ? genericArray : typeof b.valueOf !== "function" && typeof b.toString !== "function" || isNaN(b) ? object : interpolateNumber)(a, b);
}
function interpolateRound(a, b) {
  return a = +a, b = +b, function(t) {
    return Math.round(a * (1 - t) + b * t);
  };
}
var degrees = 180 / Math.PI;
var identity$2 = {
  translateX: 0,
  translateY: 0,
  rotate: 0,
  skewX: 0,
  scaleX: 1,
  scaleY: 1
};
function decompose(a, b, c, d, e, f) {
  var scaleX, scaleY, skewX;
  if (scaleX = Math.sqrt(a * a + b * b))
    a /= scaleX, b /= scaleX;
  if (skewX = a * c + b * d)
    c -= a * skewX, d -= b * skewX;
  if (scaleY = Math.sqrt(c * c + d * d))
    c /= scaleY, d /= scaleY, skewX /= scaleY;
  if (a * d < b * c)
    a = -a, b = -b, skewX = -skewX, scaleX = -scaleX;
  return {
    translateX: e,
    translateY: f,
    rotate: Math.atan2(b, a) * degrees,
    skewX: Math.atan(skewX) * degrees,
    scaleX,
    scaleY
  };
}
var svgNode;
function parseCss(value) {
  const m = new (typeof DOMMatrix === "function" ? DOMMatrix : WebKitCSSMatrix)(value + "");
  return m.isIdentity ? identity$2 : decompose(m.a, m.b, m.c, m.d, m.e, m.f);
}
function parseSvg(value) {
  if (value == null)
    return identity$2;
  if (!svgNode)
    svgNode = document.createElementNS("http://www.w3.org/2000/svg", "g");
  svgNode.setAttribute("transform", value);
  if (!(value = svgNode.transform.baseVal.consolidate()))
    return identity$2;
  value = value.matrix;
  return decompose(value.a, value.b, value.c, value.d, value.e, value.f);
}
function interpolateTransform(parse, pxComma, pxParen, degParen) {
  function pop(s) {
    return s.length ? s.pop() + " " : "";
  }
  function translate2(xa, ya, xb, yb, s, q) {
    if (xa !== xb || ya !== yb) {
      var i2 = s.push("translate(", null, pxComma, null, pxParen);
      q.push({ i: i2 - 4, x: interpolateNumber(xa, xb) }, { i: i2 - 2, x: interpolateNumber(ya, yb) });
    } else if (xb || yb) {
      s.push("translate(" + xb + pxComma + yb + pxParen);
    }
  }
  function rotate(a, b, s, q) {
    if (a !== b) {
      if (a - b > 180)
        b += 360;
      else if (b - a > 180)
        a += 360;
      q.push({ i: s.push(pop(s) + "rotate(", null, degParen) - 2, x: interpolateNumber(a, b) });
    } else if (b) {
      s.push(pop(s) + "rotate(" + b + degParen);
    }
  }
  function skewX(a, b, s, q) {
    if (a !== b) {
      q.push({ i: s.push(pop(s) + "skewX(", null, degParen) - 2, x: interpolateNumber(a, b) });
    } else if (b) {
      s.push(pop(s) + "skewX(" + b + degParen);
    }
  }
  function scale(xa, ya, xb, yb, s, q) {
    if (xa !== xb || ya !== yb) {
      var i2 = s.push(pop(s) + "scale(", null, ",", null, ")");
      q.push({ i: i2 - 4, x: interpolateNumber(xa, xb) }, { i: i2 - 2, x: interpolateNumber(ya, yb) });
    } else if (xb !== 1 || yb !== 1) {
      s.push(pop(s) + "scale(" + xb + "," + yb + ")");
    }
  }
  return function(a, b) {
    var s = [], q = [];
    a = parse(a), b = parse(b);
    translate2(a.translateX, a.translateY, b.translateX, b.translateY, s, q);
    rotate(a.rotate, b.rotate, s, q);
    skewX(a.skewX, b.skewX, s, q);
    scale(a.scaleX, a.scaleY, b.scaleX, b.scaleY, s, q);
    a = b = null;
    return function(t) {
      var i2 = -1, n = q.length, o;
      while (++i2 < n)
        s[(o = q[i2]).i] = o.x(t);
      return s.join("");
    };
  };
}
var interpolateTransformCss = interpolateTransform(parseCss, "px, ", "px)", "deg)");
var interpolateTransformSvg = interpolateTransform(parseSvg, ", ", ")", ")");
var frame = 0, timeout$1 = 0, interval = 0, pokeDelay = 1e3, taskHead, taskTail, clockLast = 0, clockNow = 0, clockSkew = 0, clock = typeof performance === "object" && performance.now ? performance : Date, setFrame = typeof window === "object" && window.requestAnimationFrame ? window.requestAnimationFrame.bind(window) : function(f) {
  setTimeout(f, 17);
};
function now() {
  return clockNow || (setFrame(clearNow), clockNow = clock.now() + clockSkew);
}
function clearNow() {
  clockNow = 0;
}
function Timer() {
  this._call = this._time = this._next = null;
}
Timer.prototype = timer$1.prototype = {
  constructor: Timer,
  restart: function(callback, delay3, time2) {
    if (typeof callback !== "function")
      throw new TypeError("callback is not a function");
    time2 = (time2 == null ? now() : +time2) + (delay3 == null ? 0 : +delay3);
    if (!this._next && taskTail !== this) {
      if (taskTail)
        taskTail._next = this;
      else
        taskHead = this;
      taskTail = this;
    }
    this._call = callback;
    this._time = time2;
    sleep$1();
  },
  stop: function() {
    if (this._call) {
      this._call = null;
      this._time = Infinity;
      sleep$1();
    }
  }
};
function timer$1(callback, delay3, time2) {
  var t = new Timer();
  t.restart(callback, delay3, time2);
  return t;
}
function timerFlush() {
  now();
  ++frame;
  var t = taskHead, e;
  while (t) {
    if ((e = clockNow - t._time) >= 0)
      t._call.call(void 0, e);
    t = t._next;
  }
  --frame;
}
function wake() {
  clockNow = (clockLast = clock.now()) + clockSkew;
  frame = timeout$1 = 0;
  try {
    timerFlush();
  } finally {
    frame = 0;
    nap();
    clockNow = 0;
  }
}
function poke() {
  var now2 = clock.now(), delay3 = now2 - clockLast;
  if (delay3 > pokeDelay)
    clockSkew -= delay3, clockLast = now2;
}
function nap() {
  var t0, t1 = taskHead, t2, time2 = Infinity;
  while (t1) {
    if (t1._call) {
      if (time2 > t1._time)
        time2 = t1._time;
      t0 = t1, t1 = t1._next;
    } else {
      t2 = t1._next, t1._next = null;
      t1 = t0 ? t0._next = t2 : taskHead = t2;
    }
  }
  taskTail = t0;
  sleep$1(time2);
}
function sleep$1(time2) {
  if (frame)
    return;
  if (timeout$1)
    timeout$1 = clearTimeout(timeout$1);
  var delay3 = time2 - clockNow;
  if (delay3 > 24) {
    if (time2 < Infinity)
      timeout$1 = setTimeout(wake, time2 - clock.now() - clockSkew);
    if (interval)
      interval = clearInterval(interval);
  } else {
    if (!interval)
      clockLast = clock.now(), interval = setInterval(poke, pokeDelay);
    frame = 1, setFrame(wake);
  }
}
function timeout(callback, delay3, time2) {
  var t = new Timer();
  delay3 = delay3 == null ? 0 : +delay3;
  t.restart((elapsed) => {
    t.stop();
    callback(elapsed + delay3);
  }, delay3, time2);
  return t;
}
var emptyOn = dispatch("start", "end", "cancel", "interrupt");
var emptyTween = [];
var CREATED = 0;
var SCHEDULED = 1;
var STARTING = 2;
var STARTED = 3;
var RUNNING = 4;
var ENDING = 5;
var ENDED = 6;
function schedule(node, name2, id2, index, group2, timing) {
  var schedules = node.__transition;
  if (!schedules)
    node.__transition = {};
  else if (id2 in schedules)
    return;
  create(node, id2, {
    name: name2,
    index,
    // For context during callback.
    group: group2,
    // For context during callback.
    on: emptyOn,
    tween: emptyTween,
    time: timing.time,
    delay: timing.delay,
    duration: timing.duration,
    ease: timing.ease,
    timer: null,
    state: CREATED
  });
}
function init(node, id2) {
  var schedule2 = get(node, id2);
  if (schedule2.state > CREATED)
    throw new Error("too late; already scheduled");
  return schedule2;
}
function set(node, id2) {
  var schedule2 = get(node, id2);
  if (schedule2.state > STARTED)
    throw new Error("too late; already running");
  return schedule2;
}
function get(node, id2) {
  var schedule2 = node.__transition;
  if (!schedule2 || !(schedule2 = schedule2[id2]))
    throw new Error("transition not found");
  return schedule2;
}
function create(node, id2, self) {
  var schedules = node.__transition, tween;
  schedules[id2] = self;
  self.timer = timer$1(schedule2, 0, self.time);
  function schedule2(elapsed) {
    self.state = SCHEDULED;
    self.timer.restart(start2, self.delay, self.time);
    if (self.delay <= elapsed)
      start2(elapsed - self.delay);
  }
  function start2(elapsed) {
    var i2, j, n, o;
    if (self.state !== SCHEDULED)
      return stop2();
    for (i2 in schedules) {
      o = schedules[i2];
      if (o.name !== self.name)
        continue;
      if (o.state === STARTED)
        return timeout(start2);
      if (o.state === RUNNING) {
        o.state = ENDED;
        o.timer.stop();
        o.on.call("interrupt", node, node.__data__, o.index, o.group);
        delete schedules[i2];
      } else if (+i2 < id2) {
        o.state = ENDED;
        o.timer.stop();
        o.on.call("cancel", node, node.__data__, o.index, o.group);
        delete schedules[i2];
      }
    }
    timeout(function() {
      if (self.state === STARTED) {
        self.state = RUNNING;
        self.timer.restart(tick2, self.delay, self.time);
        tick2(elapsed);
      }
    });
    self.state = STARTING;
    self.on.call("start", node, node.__data__, self.index, self.group);
    if (self.state !== STARTING)
      return;
    self.state = STARTED;
    tween = new Array(n = self.tween.length);
    for (i2 = 0, j = -1; i2 < n; ++i2) {
      if (o = self.tween[i2].value.call(node, node.__data__, self.index, self.group)) {
        tween[++j] = o;
      }
    }
    tween.length = j + 1;
  }
  function tick2(elapsed) {
    var t = elapsed < self.duration ? self.ease.call(null, elapsed / self.duration) : (self.timer.restart(stop2), self.state = ENDING, 1), i2 = -1, n = tween.length;
    while (++i2 < n) {
      tween[i2].call(node, t);
    }
    if (self.state === ENDING) {
      self.on.call("end", node, node.__data__, self.index, self.group);
      stop2();
    }
  }
  function stop2() {
    self.state = ENDED;
    self.timer.stop();
    delete schedules[id2];
    for (var i2 in schedules)
      return;
    delete node.__transition;
  }
}
function interrupt(node, name2) {
  var schedules = node.__transition, schedule2, active, empty2 = true, i2;
  if (!schedules)
    return;
  name2 = name2 == null ? null : name2 + "";
  for (i2 in schedules) {
    if ((schedule2 = schedules[i2]).name !== name2) {
      empty2 = false;
      continue;
    }
    active = schedule2.state > STARTING && schedule2.state < ENDING;
    schedule2.state = ENDED;
    schedule2.timer.stop();
    schedule2.on.call(active ? "interrupt" : "cancel", node, node.__data__, schedule2.index, schedule2.group);
    delete schedules[i2];
  }
  if (empty2)
    delete node.__transition;
}
function selection_interrupt(name2) {
  return this.each(function() {
    interrupt(this, name2);
  });
}
function tweenRemove(id2, name2) {
  var tween0, tween1;
  return function() {
    var schedule2 = set(this, id2), tween = schedule2.tween;
    if (tween !== tween0) {
      tween1 = tween0 = tween;
      for (var i2 = 0, n = tween1.length; i2 < n; ++i2) {
        if (tween1[i2].name === name2) {
          tween1 = tween1.slice();
          tween1.splice(i2, 1);
          break;
        }
      }
    }
    schedule2.tween = tween1;
  };
}
function tweenFunction(id2, name2, value) {
  var tween0, tween1;
  if (typeof value !== "function")
    throw new Error();
  return function() {
    var schedule2 = set(this, id2), tween = schedule2.tween;
    if (tween !== tween0) {
      tween1 = (tween0 = tween).slice();
      for (var t = { name: name2, value }, i2 = 0, n = tween1.length; i2 < n; ++i2) {
        if (tween1[i2].name === name2) {
          tween1[i2] = t;
          break;
        }
      }
      if (i2 === n)
        tween1.push(t);
    }
    schedule2.tween = tween1;
  };
}
function transition_tween(name2, value) {
  var id2 = this._id;
  name2 += "";
  if (arguments.length < 2) {
    var tween = get(this.node(), id2).tween;
    for (var i2 = 0, n = tween.length, t; i2 < n; ++i2) {
      if ((t = tween[i2]).name === name2) {
        return t.value;
      }
    }
    return null;
  }
  return this.each((value == null ? tweenRemove : tweenFunction)(id2, name2, value));
}
function tweenValue(transition, name2, value) {
  var id2 = transition._id;
  transition.each(function() {
    var schedule2 = set(this, id2);
    (schedule2.value || (schedule2.value = {}))[name2] = value.apply(this, arguments);
  });
  return function(node) {
    return get(node, id2).value[name2];
  };
}
function interpolate(a, b) {
  var c;
  return (typeof b === "number" ? interpolateNumber : b instanceof color ? interpolateRgb : (c = color(b)) ? (b = c, interpolateRgb) : interpolateString)(a, b);
}
function attrRemove(name2) {
  return function() {
    this.removeAttribute(name2);
  };
}
function attrRemoveNS(fullname) {
  return function() {
    this.removeAttributeNS(fullname.space, fullname.local);
  };
}
function attrConstant(name2, interpolate2, value1) {
  var string00, string1 = value1 + "", interpolate0;
  return function() {
    var string0 = this.getAttribute(name2);
    return string0 === string1 ? null : string0 === string00 ? interpolate0 : interpolate0 = interpolate2(string00 = string0, value1);
  };
}
function attrConstantNS(fullname, interpolate2, value1) {
  var string00, string1 = value1 + "", interpolate0;
  return function() {
    var string0 = this.getAttributeNS(fullname.space, fullname.local);
    return string0 === string1 ? null : string0 === string00 ? interpolate0 : interpolate0 = interpolate2(string00 = string0, value1);
  };
}
function attrFunction(name2, interpolate2, value) {
  var string00, string10, interpolate0;
  return function() {
    var string0, value1 = value(this), string1;
    if (value1 == null)
      return void this.removeAttribute(name2);
    string0 = this.getAttribute(name2);
    string1 = value1 + "";
    return string0 === string1 ? null : string0 === string00 && string1 === string10 ? interpolate0 : (string10 = string1, interpolate0 = interpolate2(string00 = string0, value1));
  };
}
function attrFunctionNS(fullname, interpolate2, value) {
  var string00, string10, interpolate0;
  return function() {
    var string0, value1 = value(this), string1;
    if (value1 == null)
      return void this.removeAttributeNS(fullname.space, fullname.local);
    string0 = this.getAttributeNS(fullname.space, fullname.local);
    string1 = value1 + "";
    return string0 === string1 ? null : string0 === string00 && string1 === string10 ? interpolate0 : (string10 = string1, interpolate0 = interpolate2(string00 = string0, value1));
  };
}
function transition_attr(name2, value) {
  var fullname = namespace(name2), i2 = fullname === "transform" ? interpolateTransformSvg : interpolate;
  return this.attrTween(name2, typeof value === "function" ? (fullname.local ? attrFunctionNS : attrFunction)(fullname, i2, tweenValue(this, "attr." + name2, value)) : value == null ? (fullname.local ? attrRemoveNS : attrRemove)(fullname) : (fullname.local ? attrConstantNS : attrConstant)(fullname, i2, value));
}
function attrInterpolate(name2, i2) {
  return function(t) {
    this.setAttribute(name2, i2.call(this, t));
  };
}
function attrInterpolateNS(fullname, i2) {
  return function(t) {
    this.setAttributeNS(fullname.space, fullname.local, i2.call(this, t));
  };
}
function attrTweenNS(fullname, value) {
  var t0, i0;
  function tween() {
    var i2 = value.apply(this, arguments);
    if (i2 !== i0)
      t0 = (i0 = i2) && attrInterpolateNS(fullname, i2);
    return t0;
  }
  tween._value = value;
  return tween;
}
function attrTween(name2, value) {
  var t0, i0;
  function tween() {
    var i2 = value.apply(this, arguments);
    if (i2 !== i0)
      t0 = (i0 = i2) && attrInterpolate(name2, i2);
    return t0;
  }
  tween._value = value;
  return tween;
}
function transition_attrTween(name2, value) {
  var key2 = "attr." + name2;
  if (arguments.length < 2)
    return (key2 = this.tween(key2)) && key2._value;
  if (value == null)
    return this.tween(key2, null);
  if (typeof value !== "function")
    throw new Error();
  var fullname = namespace(name2);
  return this.tween(key2, (fullname.local ? attrTweenNS : attrTween)(fullname, value));
}
function delayFunction(id2, value) {
  return function() {
    init(this, id2).delay = +value.apply(this, arguments);
  };
}
function delayConstant(id2, value) {
  return value = +value, function() {
    init(this, id2).delay = value;
  };
}
function transition_delay(value) {
  var id2 = this._id;
  return arguments.length ? this.each((typeof value === "function" ? delayFunction : delayConstant)(id2, value)) : get(this.node(), id2).delay;
}
function durationFunction(id2, value) {
  return function() {
    set(this, id2).duration = +value.apply(this, arguments);
  };
}
function durationConstant(id2, value) {
  return value = +value, function() {
    set(this, id2).duration = value;
  };
}
function transition_duration(value) {
  var id2 = this._id;
  return arguments.length ? this.each((typeof value === "function" ? durationFunction : durationConstant)(id2, value)) : get(this.node(), id2).duration;
}
function easeConstant(id2, value) {
  if (typeof value !== "function")
    throw new Error();
  return function() {
    set(this, id2).ease = value;
  };
}
function transition_ease(value) {
  var id2 = this._id;
  return arguments.length ? this.each(easeConstant(id2, value)) : get(this.node(), id2).ease;
}
function easeVarying(id2, value) {
  return function() {
    var v = value.apply(this, arguments);
    if (typeof v !== "function")
      throw new Error();
    set(this, id2).ease = v;
  };
}
function transition_easeVarying(value) {
  if (typeof value !== "function")
    throw new Error();
  return this.each(easeVarying(this._id, value));
}
function transition_filter(match) {
  if (typeof match !== "function")
    match = matcher(match);
  for (var groups = this._groups, m = groups.length, subgroups = new Array(m), j = 0; j < m; ++j) {
    for (var group2 = groups[j], n = group2.length, subgroup = subgroups[j] = [], node, i2 = 0; i2 < n; ++i2) {
      if ((node = group2[i2]) && match.call(node, node.__data__, i2, group2)) {
        subgroup.push(node);
      }
    }
  }
  return new Transition(subgroups, this._parents, this._name, this._id);
}
function transition_merge(transition) {
  if (transition._id !== this._id)
    throw new Error();
  for (var groups0 = this._groups, groups1 = transition._groups, m0 = groups0.length, m1 = groups1.length, m = Math.min(m0, m1), merges = new Array(m0), j = 0; j < m; ++j) {
    for (var group0 = groups0[j], group1 = groups1[j], n = group0.length, merge = merges[j] = new Array(n), node, i2 = 0; i2 < n; ++i2) {
      if (node = group0[i2] || group1[i2]) {
        merge[i2] = node;
      }
    }
  }
  for (; j < m0; ++j) {
    merges[j] = groups0[j];
  }
  return new Transition(merges, this._parents, this._name, this._id);
}
function start(name2) {
  return (name2 + "").trim().split(/^|\s+/).every(function(t) {
    var i2 = t.indexOf(".");
    if (i2 >= 0)
      t = t.slice(0, i2);
    return !t || t === "start";
  });
}
function onFunction(id2, name2, listener) {
  var on0, on1, sit = start(name2) ? init : set;
  return function() {
    var schedule2 = sit(this, id2), on = schedule2.on;
    if (on !== on0)
      (on1 = (on0 = on).copy()).on(name2, listener);
    schedule2.on = on1;
  };
}
function transition_on(name2, listener) {
  var id2 = this._id;
  return arguments.length < 2 ? get(this.node(), id2).on.on(name2) : this.each(onFunction(id2, name2, listener));
}
function removeFunction(id2) {
  return function() {
    var parent = this.parentNode;
    for (var i2 in this.__transition)
      if (+i2 !== id2)
        return;
    if (parent)
      parent.removeChild(this);
  };
}
function transition_remove() {
  return this.on("end.remove", removeFunction(this._id));
}
function transition_select(select2) {
  var name2 = this._name, id2 = this._id;
  if (typeof select2 !== "function")
    select2 = selector(select2);
  for (var groups = this._groups, m = groups.length, subgroups = new Array(m), j = 0; j < m; ++j) {
    for (var group2 = groups[j], n = group2.length, subgroup = subgroups[j] = new Array(n), node, subnode, i2 = 0; i2 < n; ++i2) {
      if ((node = group2[i2]) && (subnode = select2.call(node, node.__data__, i2, group2))) {
        if ("__data__" in node)
          subnode.__data__ = node.__data__;
        subgroup[i2] = subnode;
        schedule(subgroup[i2], name2, id2, i2, subgroup, get(node, id2));
      }
    }
  }
  return new Transition(subgroups, this._parents, name2, id2);
}
function transition_selectAll(select2) {
  var name2 = this._name, id2 = this._id;
  if (typeof select2 !== "function")
    select2 = selectorAll(select2);
  for (var groups = this._groups, m = groups.length, subgroups = [], parents = [], j = 0; j < m; ++j) {
    for (var group2 = groups[j], n = group2.length, node, i2 = 0; i2 < n; ++i2) {
      if (node = group2[i2]) {
        for (var children2 = select2.call(node, node.__data__, i2, group2), child, inherit2 = get(node, id2), k = 0, l = children2.length; k < l; ++k) {
          if (child = children2[k]) {
            schedule(child, name2, id2, k, children2, inherit2);
          }
        }
        subgroups.push(children2);
        parents.push(node);
      }
    }
  }
  return new Transition(subgroups, parents, name2, id2);
}
var Selection = selection.prototype.constructor;
function transition_selection() {
  return new Selection(this._groups, this._parents);
}
function styleNull(name2, interpolate2) {
  var string00, string10, interpolate0;
  return function() {
    var string0 = styleValue(this, name2), string1 = (this.style.removeProperty(name2), styleValue(this, name2));
    return string0 === string1 ? null : string0 === string00 && string1 === string10 ? interpolate0 : interpolate0 = interpolate2(string00 = string0, string10 = string1);
  };
}
function styleRemove(name2) {
  return function() {
    this.style.removeProperty(name2);
  };
}
function styleConstant(name2, interpolate2, value1) {
  var string00, string1 = value1 + "", interpolate0;
  return function() {
    var string0 = styleValue(this, name2);
    return string0 === string1 ? null : string0 === string00 ? interpolate0 : interpolate0 = interpolate2(string00 = string0, value1);
  };
}
function styleFunction(name2, interpolate2, value) {
  var string00, string10, interpolate0;
  return function() {
    var string0 = styleValue(this, name2), value1 = value(this), string1 = value1 + "";
    if (value1 == null)
      string1 = value1 = (this.style.removeProperty(name2), styleValue(this, name2));
    return string0 === string1 ? null : string0 === string00 && string1 === string10 ? interpolate0 : (string10 = string1, interpolate0 = interpolate2(string00 = string0, value1));
  };
}
function styleMaybeRemove(id2, name2) {
  var on0, on1, listener0, key2 = "style." + name2, event = "end." + key2, remove2;
  return function() {
    var schedule2 = set(this, id2), on = schedule2.on, listener = schedule2.value[key2] == null ? remove2 || (remove2 = styleRemove(name2)) : void 0;
    if (on !== on0 || listener0 !== listener)
      (on1 = (on0 = on).copy()).on(event, listener0 = listener);
    schedule2.on = on1;
  };
}
function transition_style(name2, value, priority) {
  var i2 = (name2 += "") === "transform" ? interpolateTransformCss : interpolate;
  return value == null ? this.styleTween(name2, styleNull(name2, i2)).on("end.style." + name2, styleRemove(name2)) : typeof value === "function" ? this.styleTween(name2, styleFunction(name2, i2, tweenValue(this, "style." + name2, value))).each(styleMaybeRemove(this._id, name2)) : this.styleTween(name2, styleConstant(name2, i2, value), priority).on("end.style." + name2, null);
}
function styleInterpolate(name2, i2, priority) {
  return function(t) {
    this.style.setProperty(name2, i2.call(this, t), priority);
  };
}
function styleTween(name2, value, priority) {
  var t, i0;
  function tween() {
    var i2 = value.apply(this, arguments);
    if (i2 !== i0)
      t = (i0 = i2) && styleInterpolate(name2, i2, priority);
    return t;
  }
  tween._value = value;
  return tween;
}
function transition_styleTween(name2, value, priority) {
  var key2 = "style." + (name2 += "");
  if (arguments.length < 2)
    return (key2 = this.tween(key2)) && key2._value;
  if (value == null)
    return this.tween(key2, null);
  if (typeof value !== "function")
    throw new Error();
  return this.tween(key2, styleTween(name2, value, priority == null ? "" : priority));
}
function textConstant(value) {
  return function() {
    this.textContent = value;
  };
}
function textFunction(value) {
  return function() {
    var value1 = value(this);
    this.textContent = value1 == null ? "" : value1;
  };
}
function transition_text(value) {
  return this.tween("text", typeof value === "function" ? textFunction(tweenValue(this, "text", value)) : textConstant(value == null ? "" : value + ""));
}
function textInterpolate(i2) {
  return function(t) {
    this.textContent = i2.call(this, t);
  };
}
function textTween(value) {
  var t0, i0;
  function tween() {
    var i2 = value.apply(this, arguments);
    if (i2 !== i0)
      t0 = (i0 = i2) && textInterpolate(i2);
    return t0;
  }
  tween._value = value;
  return tween;
}
function transition_textTween(value) {
  var key2 = "text";
  if (arguments.length < 1)
    return (key2 = this.tween(key2)) && key2._value;
  if (value == null)
    return this.tween(key2, null);
  if (typeof value !== "function")
    throw new Error();
  return this.tween(key2, textTween(value));
}
function transition_transition() {
  var name2 = this._name, id0 = this._id, id1 = newId();
  for (var groups = this._groups, m = groups.length, j = 0; j < m; ++j) {
    for (var group2 = groups[j], n = group2.length, node, i2 = 0; i2 < n; ++i2) {
      if (node = group2[i2]) {
        var inherit2 = get(node, id0);
        schedule(node, name2, id1, i2, group2, {
          time: inherit2.time + inherit2.delay + inherit2.duration,
          delay: 0,
          duration: inherit2.duration,
          ease: inherit2.ease
        });
      }
    }
  }
  return new Transition(groups, this._parents, name2, id1);
}
function transition_end() {
  var on0, on1, that = this, id2 = that._id, size2 = that.size();
  return new Promise(function(resolve, reject) {
    var cancel = { value: reject }, end = { value: function() {
      if (--size2 === 0)
        resolve();
    } };
    that.each(function() {
      var schedule2 = set(this, id2), on = schedule2.on;
      if (on !== on0) {
        on1 = (on0 = on).copy();
        on1._.cancel.push(cancel);
        on1._.interrupt.push(cancel);
        on1._.end.push(end);
      }
      schedule2.on = on1;
    });
    if (size2 === 0)
      resolve();
  });
}
var id = 0;
function Transition(groups, parents, name2, id2) {
  this._groups = groups;
  this._parents = parents;
  this._name = name2;
  this._id = id2;
}
function newId() {
  return ++id;
}
var selection_prototype = selection.prototype;
Transition.prototype = {
  constructor: Transition,
  select: transition_select,
  selectAll: transition_selectAll,
  selectChild: selection_prototype.selectChild,
  selectChildren: selection_prototype.selectChildren,
  filter: transition_filter,
  merge: transition_merge,
  selection: transition_selection,
  transition: transition_transition,
  call: selection_prototype.call,
  nodes: selection_prototype.nodes,
  node: selection_prototype.node,
  size: selection_prototype.size,
  empty: selection_prototype.empty,
  each: selection_prototype.each,
  on: transition_on,
  attr: transition_attr,
  attrTween: transition_attrTween,
  style: transition_style,
  styleTween: transition_styleTween,
  text: transition_text,
  textTween: transition_textTween,
  remove: transition_remove,
  tween: transition_tween,
  delay: transition_delay,
  duration: transition_duration,
  ease: transition_ease,
  easeVarying: transition_easeVarying,
  end: transition_end,
  [Symbol.iterator]: selection_prototype[Symbol.iterator]
};
function cubicInOut(t) {
  return ((t *= 2) <= 1 ? t * t * t : (t -= 2) * t * t + 2) / 2;
}
var defaultTiming = {
  time: null,
  // Set on use.
  delay: 0,
  duration: 250,
  ease: cubicInOut
};
function inherit(node, id2) {
  var timing;
  while (!(timing = node.__transition) || !(timing = timing[id2])) {
    if (!(node = node.parentNode)) {
      throw new Error(`transition ${id2} not found`);
    }
  }
  return timing;
}
function selection_transition(name2) {
  var id2, timing;
  if (name2 instanceof Transition) {
    id2 = name2._id, name2 = name2._name;
  } else {
    id2 = newId(), (timing = defaultTiming).time = now(), name2 = name2 == null ? null : name2 + "";
  }
  for (var groups = this._groups, m = groups.length, j = 0; j < m; ++j) {
    for (var group2 = groups[j], n = group2.length, node, i2 = 0; i2 < n; ++i2) {
      if (node = group2[i2]) {
        schedule(node, name2, id2, i2, group2, timing || inherit(node, id2));
      }
    }
  }
  return new Transition(groups, this._parents, name2, id2);
}
selection.prototype.interrupt = selection_interrupt;
selection.prototype.transition = selection_transition;
const pi = Math.PI, tau = 2 * pi, epsilon = 1e-6, tauEpsilon = tau - epsilon;
function append(strings) {
  this._ += strings[0];
  for (let i2 = 1, n = strings.length; i2 < n; ++i2) {
    this._ += arguments[i2] + strings[i2];
  }
}
function appendRound(digits) {
  let d = Math.floor(digits);
  if (!(d >= 0))
    throw new Error(`invalid digits: ${digits}`);
  if (d > 15)
    return append;
  const k = 10 ** d;
  return function(strings) {
    this._ += strings[0];
    for (let i2 = 1, n = strings.length; i2 < n; ++i2) {
      this._ += Math.round(arguments[i2] * k) / k + strings[i2];
    }
  };
}
class Path {
  constructor(digits) {
    this._x0 = this._y0 = // start of current subpath
    this._x1 = this._y1 = null;
    this._ = "";
    this._append = digits == null ? append : appendRound(digits);
  }
  moveTo(x2, y2) {
    this._append`M${this._x0 = this._x1 = +x2},${this._y0 = this._y1 = +y2}`;
  }
  closePath() {
    if (this._x1 !== null) {
      this._x1 = this._x0, this._y1 = this._y0;
      this._append`Z`;
    }
  }
  lineTo(x2, y2) {
    this._append`L${this._x1 = +x2},${this._y1 = +y2}`;
  }
  quadraticCurveTo(x1, y1, x2, y2) {
    this._append`Q${+x1},${+y1},${this._x1 = +x2},${this._y1 = +y2}`;
  }
  bezierCurveTo(x1, y1, x2, y2, x3, y3) {
    this._append`C${+x1},${+y1},${+x2},${+y2},${this._x1 = +x3},${this._y1 = +y3}`;
  }
  arcTo(x1, y1, x2, y2, r) {
    x1 = +x1, y1 = +y1, x2 = +x2, y2 = +y2, r = +r;
    if (r < 0)
      throw new Error(`negative radius: ${r}`);
    let x0 = this._x1, y0 = this._y1, x21 = x2 - x1, y21 = y2 - y1, x01 = x0 - x1, y01 = y0 - y1, l01_2 = x01 * x01 + y01 * y01;
    if (this._x1 === null) {
      this._append`M${this._x1 = x1},${this._y1 = y1}`;
    } else if (!(l01_2 > epsilon))
      ;
    else if (!(Math.abs(y01 * x21 - y21 * x01) > epsilon) || !r) {
      this._append`L${this._x1 = x1},${this._y1 = y1}`;
    } else {
      let x20 = x2 - x0, y20 = y2 - y0, l21_2 = x21 * x21 + y21 * y21, l20_2 = x20 * x20 + y20 * y20, l21 = Math.sqrt(l21_2), l01 = Math.sqrt(l01_2), l = r * Math.tan((pi - Math.acos((l21_2 + l01_2 - l20_2) / (2 * l21 * l01))) / 2), t01 = l / l01, t21 = l / l21;
      if (Math.abs(t01 - 1) > epsilon) {
        this._append`L${x1 + t01 * x01},${y1 + t01 * y01}`;
      }
      this._append`A${r},${r},0,0,${+(y01 * x20 > x01 * y20)},${this._x1 = x1 + t21 * x21},${this._y1 = y1 + t21 * y21}`;
    }
  }
  arc(x2, y2, r, a0, a1, ccw) {
    x2 = +x2, y2 = +y2, r = +r, ccw = !!ccw;
    if (r < 0)
      throw new Error(`negative radius: ${r}`);
    let dx = r * Math.cos(a0), dy = r * Math.sin(a0), x0 = x2 + dx, y0 = y2 + dy, cw = 1 ^ ccw, da = ccw ? a0 - a1 : a1 - a0;
    if (this._x1 === null) {
      this._append`M${x0},${y0}`;
    } else if (Math.abs(this._x1 - x0) > epsilon || Math.abs(this._y1 - y0) > epsilon) {
      this._append`L${x0},${y0}`;
    }
    if (!r)
      return;
    if (da < 0)
      da = da % tau + tau;
    if (da > tauEpsilon) {
      this._append`A${r},${r},0,1,${cw},${x2 - dx},${y2 - dy}A${r},${r},0,1,${cw},${this._x1 = x0},${this._y1 = y0}`;
    } else if (da > epsilon) {
      this._append`A${r},${r},0,${+(da >= pi)},${cw},${this._x1 = x2 + r * Math.cos(a1)},${this._y1 = y2 + r * Math.sin(a1)}`;
    }
  }
  rect(x2, y2, w, h) {
    this._append`M${this._x0 = this._x1 = +x2},${this._y0 = this._y1 = +y2}h${w = +w}v${+h}h${-w}Z`;
  }
  toString() {
    return this._;
  }
}
function formatDecimal(x2) {
  return Math.abs(x2 = Math.round(x2)) >= 1e21 ? x2.toLocaleString("en").replace(/,/g, "") : x2.toString(10);
}
function formatDecimalParts(x2, p) {
  if ((i2 = (x2 = p ? x2.toExponential(p - 1) : x2.toExponential()).indexOf("e")) < 0)
    return null;
  var i2, coefficient = x2.slice(0, i2);
  return [
    coefficient.length > 1 ? coefficient[0] + coefficient.slice(2) : coefficient,
    +x2.slice(i2 + 1)
  ];
}
function exponent(x2) {
  return x2 = formatDecimalParts(Math.abs(x2)), x2 ? x2[1] : NaN;
}
function formatGroup(grouping, thousands) {
  return function(value, width) {
    var i2 = value.length, t = [], j = 0, g = grouping[0], length = 0;
    while (i2 > 0 && g > 0) {
      if (length + g + 1 > width)
        g = Math.max(1, width - length);
      t.push(value.substring(i2 -= g, i2 + g));
      if ((length += g + 1) > width)
        break;
      g = grouping[j = (j + 1) % grouping.length];
    }
    return t.reverse().join(thousands);
  };
}
function formatNumerals(numerals) {
  return function(value) {
    return value.replace(/[0-9]/g, function(i2) {
      return numerals[+i2];
    });
  };
}
var re = /^(?:(.)?([<>=^]))?([+\-( ])?([$#])?(0)?(\d+)?(,)?(\.\d+)?(~)?([a-z%])?$/i;
function formatSpecifier(specifier) {
  if (!(match = re.exec(specifier)))
    throw new Error("invalid format: " + specifier);
  var match;
  return new FormatSpecifier({
    fill: match[1],
    align: match[2],
    sign: match[3],
    symbol: match[4],
    zero: match[5],
    width: match[6],
    comma: match[7],
    precision: match[8] && match[8].slice(1),
    trim: match[9],
    type: match[10]
  });
}
formatSpecifier.prototype = FormatSpecifier.prototype;
function FormatSpecifier(specifier) {
  this.fill = specifier.fill === void 0 ? " " : specifier.fill + "";
  this.align = specifier.align === void 0 ? ">" : specifier.align + "";
  this.sign = specifier.sign === void 0 ? "-" : specifier.sign + "";
  this.symbol = specifier.symbol === void 0 ? "" : specifier.symbol + "";
  this.zero = !!specifier.zero;
  this.width = specifier.width === void 0 ? void 0 : +specifier.width;
  this.comma = !!specifier.comma;
  this.precision = specifier.precision === void 0 ? void 0 : +specifier.precision;
  this.trim = !!specifier.trim;
  this.type = specifier.type === void 0 ? "" : specifier.type + "";
}
FormatSpecifier.prototype.toString = function() {
  return this.fill + this.align + this.sign + this.symbol + (this.zero ? "0" : "") + (this.width === void 0 ? "" : Math.max(1, this.width | 0)) + (this.comma ? "," : "") + (this.precision === void 0 ? "" : "." + Math.max(0, this.precision | 0)) + (this.trim ? "~" : "") + this.type;
};
function formatTrim(s) {
  out:
    for (var n = s.length, i2 = 1, i0 = -1, i1; i2 < n; ++i2) {
      switch (s[i2]) {
        case ".":
          i0 = i1 = i2;
          break;
        case "0":
          if (i0 === 0)
            i0 = i2;
          i1 = i2;
          break;
        default:
          if (!+s[i2])
            break out;
          if (i0 > 0)
            i0 = 0;
          break;
      }
    }
  return i0 > 0 ? s.slice(0, i0) + s.slice(i1 + 1) : s;
}
var prefixExponent;
function formatPrefixAuto(x2, p) {
  var d = formatDecimalParts(x2, p);
  if (!d)
    return x2 + "";
  var coefficient = d[0], exponent2 = d[1], i2 = exponent2 - (prefixExponent = Math.max(-8, Math.min(8, Math.floor(exponent2 / 3))) * 3) + 1, n = coefficient.length;
  return i2 === n ? coefficient : i2 > n ? coefficient + new Array(i2 - n + 1).join("0") : i2 > 0 ? coefficient.slice(0, i2) + "." + coefficient.slice(i2) : "0." + new Array(1 - i2).join("0") + formatDecimalParts(x2, Math.max(0, p + i2 - 1))[0];
}
function formatRounded(x2, p) {
  var d = formatDecimalParts(x2, p);
  if (!d)
    return x2 + "";
  var coefficient = d[0], exponent2 = d[1];
  return exponent2 < 0 ? "0." + new Array(-exponent2).join("0") + coefficient : coefficient.length > exponent2 + 1 ? coefficient.slice(0, exponent2 + 1) + "." + coefficient.slice(exponent2 + 1) : coefficient + new Array(exponent2 - coefficient.length + 2).join("0");
}
const formatTypes = {
  "%": (x2, p) => (x2 * 100).toFixed(p),
  "b": (x2) => Math.round(x2).toString(2),
  "c": (x2) => x2 + "",
  "d": formatDecimal,
  "e": (x2, p) => x2.toExponential(p),
  "f": (x2, p) => x2.toFixed(p),
  "g": (x2, p) => x2.toPrecision(p),
  "o": (x2) => Math.round(x2).toString(8),
  "p": (x2, p) => formatRounded(x2 * 100, p),
  "r": formatRounded,
  "s": formatPrefixAuto,
  "X": (x2) => Math.round(x2).toString(16).toUpperCase(),
  "x": (x2) => Math.round(x2).toString(16)
};
function identity$1(x2) {
  return x2;
}
var map$1 = Array.prototype.map, prefixes = ["y", "z", "a", "f", "p", "n", "µ", "m", "", "k", "M", "G", "T", "P", "E", "Z", "Y"];
function formatLocale(locale2) {
  var group2 = locale2.grouping === void 0 || locale2.thousands === void 0 ? identity$1 : formatGroup(map$1.call(locale2.grouping, Number), locale2.thousands + ""), currencyPrefix = locale2.currency === void 0 ? "" : locale2.currency[0] + "", currencySuffix = locale2.currency === void 0 ? "" : locale2.currency[1] + "", decimal = locale2.decimal === void 0 ? "." : locale2.decimal + "", numerals = locale2.numerals === void 0 ? identity$1 : formatNumerals(map$1.call(locale2.numerals, String)), percent = locale2.percent === void 0 ? "%" : locale2.percent + "", minus = locale2.minus === void 0 ? "−" : locale2.minus + "", nan = locale2.nan === void 0 ? "NaN" : locale2.nan + "";
  function newFormat(specifier) {
    specifier = formatSpecifier(specifier);
    var fill = specifier.fill, align = specifier.align, sign = specifier.sign, symbol = specifier.symbol, zero2 = specifier.zero, width = specifier.width, comma = specifier.comma, precision = specifier.precision, trim = specifier.trim, type = specifier.type;
    if (type === "n")
      comma = true, type = "g";
    else if (!formatTypes[type])
      precision === void 0 && (precision = 12), trim = true, type = "g";
    if (zero2 || fill === "0" && align === "=")
      zero2 = true, fill = "0", align = "=";
    var prefix = symbol === "$" ? currencyPrefix : symbol === "#" && /[boxX]/.test(type) ? "0" + type.toLowerCase() : "", suffix = symbol === "$" ? currencySuffix : /[%p]/.test(type) ? percent : "";
    var formatType = formatTypes[type], maybeSuffix = /[defgprs%]/.test(type);
    precision = precision === void 0 ? 6 : /[gprs]/.test(type) ? Math.max(1, Math.min(21, precision)) : Math.max(0, Math.min(20, precision));
    function format2(value) {
      var valuePrefix = prefix, valueSuffix = suffix, i2, n, c;
      if (type === "c") {
        valueSuffix = formatType(value) + valueSuffix;
        value = "";
      } else {
        value = +value;
        var valueNegative = value < 0 || 1 / value < 0;
        value = isNaN(value) ? nan : formatType(Math.abs(value), precision);
        if (trim)
          value = formatTrim(value);
        if (valueNegative && +value === 0 && sign !== "+")
          valueNegative = false;
        valuePrefix = (valueNegative ? sign === "(" ? sign : minus : sign === "-" || sign === "(" ? "" : sign) + valuePrefix;
        valueSuffix = (type === "s" ? prefixes[8 + prefixExponent / 3] : "") + valueSuffix + (valueNegative && sign === "(" ? ")" : "");
        if (maybeSuffix) {
          i2 = -1, n = value.length;
          while (++i2 < n) {
            if (c = value.charCodeAt(i2), 48 > c || c > 57) {
              valueSuffix = (c === 46 ? decimal + value.slice(i2 + 1) : value.slice(i2)) + valueSuffix;
              value = value.slice(0, i2);
              break;
            }
          }
        }
      }
      if (comma && !zero2)
        value = group2(value, Infinity);
      var length = valuePrefix.length + value.length + valueSuffix.length, padding = length < width ? new Array(width - length + 1).join(fill) : "";
      if (comma && zero2)
        value = group2(padding + value, padding.length ? width - valueSuffix.length : Infinity), padding = "";
      switch (align) {
        case "<":
          value = valuePrefix + value + valueSuffix + padding;
          break;
        case "=":
          value = valuePrefix + padding + value + valueSuffix;
          break;
        case "^":
          value = padding.slice(0, length = padding.length >> 1) + valuePrefix + value + valueSuffix + padding.slice(length);
          break;
        default:
          value = padding + valuePrefix + value + valueSuffix;
          break;
      }
      return numerals(value);
    }
    format2.toString = function() {
      return specifier + "";
    };
    return format2;
  }
  function formatPrefix2(specifier, value) {
    var f = newFormat((specifier = formatSpecifier(specifier), specifier.type = "f", specifier)), e = Math.max(-8, Math.min(8, Math.floor(exponent(value) / 3))) * 3, k = Math.pow(10, -e), prefix = prefixes[8 + e / 3];
    return function(value2) {
      return f(k * value2) + prefix;
    };
  }
  return {
    format: newFormat,
    formatPrefix: formatPrefix2
  };
}
var locale;
var format;
var formatPrefix;
defaultLocale({
  thousands: ",",
  grouping: [3],
  currency: ["$", ""]
});
function defaultLocale(definition) {
  locale = formatLocale(definition);
  format = locale.format;
  formatPrefix = locale.formatPrefix;
  return locale;
}
function precisionFixed(step) {
  return Math.max(0, -exponent(Math.abs(step)));
}
function precisionPrefix(step, value) {
  return Math.max(0, Math.max(-8, Math.min(8, Math.floor(exponent(value) / 3))) * 3 - exponent(Math.abs(step)));
}
function precisionRound(step, max2) {
  step = Math.abs(step), max2 = Math.abs(max2) - step;
  return Math.max(0, exponent(max2) - exponent(step)) + 1;
}
function initRange(domain, range2) {
  switch (arguments.length) {
    case 0:
      break;
    case 1:
      this.range(domain);
      break;
    default:
      this.range(range2).domain(domain);
      break;
  }
  return this;
}
const implicit = Symbol("implicit");
function ordinal() {
  var index = new InternMap(), domain = [], range2 = [], unknown = implicit;
  function scale(d) {
    let i2 = index.get(d);
    if (i2 === void 0) {
      if (unknown !== implicit)
        return unknown;
      index.set(d, i2 = domain.push(d) - 1);
    }
    return range2[i2 % range2.length];
  }
  scale.domain = function(_) {
    if (!arguments.length)
      return domain.slice();
    domain = [], index = new InternMap();
    for (const value of _) {
      if (index.has(value))
        continue;
      index.set(value, domain.push(value) - 1);
    }
    return scale;
  };
  scale.range = function(_) {
    return arguments.length ? (range2 = Array.from(_), scale) : range2.slice();
  };
  scale.unknown = function(_) {
    return arguments.length ? (unknown = _, scale) : unknown;
  };
  scale.copy = function() {
    return ordinal(domain, range2).unknown(unknown);
  };
  initRange.apply(scale, arguments);
  return scale;
}
function band() {
  var scale = ordinal().unknown(void 0), domain = scale.domain, ordinalRange = scale.range, r0 = 0, r1 = 1, step, bandwidth, round2 = false, paddingInner = 0, paddingOuter = 0, align = 0.5;
  delete scale.unknown;
  function rescale() {
    var n = domain().length, reverse = r1 < r0, start2 = reverse ? r1 : r0, stop2 = reverse ? r0 : r1;
    step = (stop2 - start2) / Math.max(1, n - paddingInner + paddingOuter * 2);
    if (round2)
      step = Math.floor(step);
    start2 += (stop2 - start2 - step * (n - paddingInner)) * align;
    bandwidth = step * (1 - paddingInner);
    if (round2)
      start2 = Math.round(start2), bandwidth = Math.round(bandwidth);
    var values = range(n).map(function(i2) {
      return start2 + step * i2;
    });
    return ordinalRange(reverse ? values.reverse() : values);
  }
  scale.domain = function(_) {
    return arguments.length ? (domain(_), rescale()) : domain();
  };
  scale.range = function(_) {
    return arguments.length ? ([r0, r1] = _, r0 = +r0, r1 = +r1, rescale()) : [r0, r1];
  };
  scale.rangeRound = function(_) {
    return [r0, r1] = _, r0 = +r0, r1 = +r1, round2 = true, rescale();
  };
  scale.bandwidth = function() {
    return bandwidth;
  };
  scale.step = function() {
    return step;
  };
  scale.round = function(_) {
    return arguments.length ? (round2 = !!_, rescale()) : round2;
  };
  scale.padding = function(_) {
    return arguments.length ? (paddingInner = Math.min(1, paddingOuter = +_), rescale()) : paddingInner;
  };
  scale.paddingInner = function(_) {
    return arguments.length ? (paddingInner = Math.min(1, _), rescale()) : paddingInner;
  };
  scale.paddingOuter = function(_) {
    return arguments.length ? (paddingOuter = +_, rescale()) : paddingOuter;
  };
  scale.align = function(_) {
    return arguments.length ? (align = Math.max(0, Math.min(1, _)), rescale()) : align;
  };
  scale.copy = function() {
    return band(domain(), [r0, r1]).round(round2).paddingInner(paddingInner).paddingOuter(paddingOuter).align(align);
  };
  return initRange.apply(rescale(), arguments);
}
function constants(x2) {
  return function() {
    return x2;
  };
}
function number(x2) {
  return +x2;
}
var unit = [0, 1];
function identity(x2) {
  return x2;
}
function normalize(a, b) {
  return (b -= a = +a) ? function(x2) {
    return (x2 - a) / b;
  } : constants(isNaN(b) ? NaN : 0.5);
}
function clamper(a, b) {
  var t;
  if (a > b)
    t = a, a = b, b = t;
  return function(x2) {
    return Math.max(a, Math.min(b, x2));
  };
}
function bimap(domain, range2, interpolate2) {
  var d0 = domain[0], d1 = domain[1], r0 = range2[0], r1 = range2[1];
  if (d1 < d0)
    d0 = normalize(d1, d0), r0 = interpolate2(r1, r0);
  else
    d0 = normalize(d0, d1), r0 = interpolate2(r0, r1);
  return function(x2) {
    return r0(d0(x2));
  };
}
function polymap(domain, range2, interpolate2) {
  var j = Math.min(domain.length, range2.length) - 1, d = new Array(j), r = new Array(j), i2 = -1;
  if (domain[j] < domain[0]) {
    domain = domain.slice().reverse();
    range2 = range2.slice().reverse();
  }
  while (++i2 < j) {
    d[i2] = normalize(domain[i2], domain[i2 + 1]);
    r[i2] = interpolate2(range2[i2], range2[i2 + 1]);
  }
  return function(x2) {
    var i3 = bisect(domain, x2, 1, j) - 1;
    return r[i3](d[i3](x2));
  };
}
function copy$1(source, target) {
  return target.domain(source.domain()).range(source.range()).interpolate(source.interpolate()).clamp(source.clamp()).unknown(source.unknown());
}
function transformer() {
  var domain = unit, range2 = unit, interpolate2 = interpolate$1, transform, untransform, unknown, clamp2 = identity, piecewise, output, input;
  function rescale() {
    var n = Math.min(domain.length, range2.length);
    if (clamp2 !== identity)
      clamp2 = clamper(domain[0], domain[n - 1]);
    piecewise = n > 2 ? polymap : bimap;
    output = input = null;
    return scale;
  }
  function scale(x2) {
    return x2 == null || isNaN(x2 = +x2) ? unknown : (output || (output = piecewise(domain.map(transform), range2, interpolate2)))(transform(clamp2(x2)));
  }
  scale.invert = function(y2) {
    return clamp2(untransform((input || (input = piecewise(range2, domain.map(transform), interpolateNumber)))(y2)));
  };
  scale.domain = function(_) {
    return arguments.length ? (domain = Array.from(_, number), rescale()) : domain.slice();
  };
  scale.range = function(_) {
    return arguments.length ? (range2 = Array.from(_), rescale()) : range2.slice();
  };
  scale.rangeRound = function(_) {
    return range2 = Array.from(_), interpolate2 = interpolateRound, rescale();
  };
  scale.clamp = function(_) {
    return arguments.length ? (clamp2 = _ ? true : identity, rescale()) : clamp2 !== identity;
  };
  scale.interpolate = function(_) {
    return arguments.length ? (interpolate2 = _, rescale()) : interpolate2;
  };
  scale.unknown = function(_) {
    return arguments.length ? (unknown = _, scale) : unknown;
  };
  return function(t, u) {
    transform = t, untransform = u;
    return rescale();
  };
}
function continuous() {
  return transformer()(identity, identity);
}
function tickFormat(start2, stop2, count, specifier) {
  var step = tickStep(start2, stop2, count), precision;
  specifier = formatSpecifier(specifier == null ? ",f" : specifier);
  switch (specifier.type) {
    case "s": {
      var value = Math.max(Math.abs(start2), Math.abs(stop2));
      if (specifier.precision == null && !isNaN(precision = precisionPrefix(step, value)))
        specifier.precision = precision;
      return formatPrefix(specifier, value);
    }
    case "":
    case "e":
    case "g":
    case "p":
    case "r": {
      if (specifier.precision == null && !isNaN(precision = precisionRound(step, Math.max(Math.abs(start2), Math.abs(stop2)))))
        specifier.precision = precision - (specifier.type === "e");
      break;
    }
    case "f":
    case "%": {
      if (specifier.precision == null && !isNaN(precision = precisionFixed(step)))
        specifier.precision = precision - (specifier.type === "%") * 2;
      break;
    }
  }
  return format(specifier);
}
function linearish(scale) {
  var domain = scale.domain;
  scale.ticks = function(count) {
    var d = domain();
    return ticks(d[0], d[d.length - 1], count == null ? 10 : count);
  };
  scale.tickFormat = function(count, specifier) {
    var d = domain();
    return tickFormat(d[0], d[d.length - 1], count == null ? 10 : count, specifier);
  };
  scale.nice = function(count) {
    if (count == null)
      count = 10;
    var d = domain();
    var i0 = 0;
    var i1 = d.length - 1;
    var start2 = d[i0];
    var stop2 = d[i1];
    var prestep;
    var step;
    var maxIter = 10;
    if (stop2 < start2) {
      step = start2, start2 = stop2, stop2 = step;
      step = i0, i0 = i1, i1 = step;
    }
    while (maxIter-- > 0) {
      step = tickIncrement(start2, stop2, count);
      if (step === prestep) {
        d[i0] = start2;
        d[i1] = stop2;
        return domain(d);
      } else if (step > 0) {
        start2 = Math.floor(start2 / step) * step;
        stop2 = Math.ceil(stop2 / step) * step;
      } else if (step < 0) {
        start2 = Math.ceil(start2 * step) / step;
        stop2 = Math.floor(stop2 * step) / step;
      } else {
        break;
      }
      prestep = step;
    }
    return scale;
  };
  return scale;
}
function linear() {
  var scale = continuous();
  scale.copy = function() {
    return copy$1(scale, linear());
  };
  initRange.apply(scale, arguments);
  return linearish(scale);
}
function constant(x2) {
  return function constant2() {
    return x2;
  };
}
function withPath(shape) {
  let digits = 3;
  shape.digits = function(_) {
    if (!arguments.length)
      return digits;
    if (_ == null) {
      digits = null;
    } else {
      const d = Math.floor(_);
      if (!(d >= 0))
        throw new RangeError(`invalid digits: ${_}`);
      digits = d;
    }
    return shape;
  };
  return () => new Path(digits);
}
function array(x2) {
  return typeof x2 === "object" && "length" in x2 ? x2 : Array.from(x2);
}
function Linear(context) {
  this._context = context;
}
Linear.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._point = 0;
  },
  lineEnd: function() {
    if (this._line || this._line !== 0 && this._point === 1)
      this._context.closePath();
    this._line = 1 - this._line;
  },
  point: function(x2, y2) {
    x2 = +x2, y2 = +y2;
    switch (this._point) {
      case 0:
        this._point = 1;
        this._line ? this._context.lineTo(x2, y2) : this._context.moveTo(x2, y2);
        break;
      case 1:
        this._point = 2;
      default:
        this._context.lineTo(x2, y2);
        break;
    }
  }
};
function curveLinear(context) {
  return new Linear(context);
}
function x(p) {
  return p[0];
}
function y(p) {
  return p[1];
}
function line$1(x$1, y$1) {
  var defined = constant(true), context = null, curve = curveLinear, output = null, path = withPath(line2);
  x$1 = typeof x$1 === "function" ? x$1 : x$1 === void 0 ? x : constant(x$1);
  y$1 = typeof y$1 === "function" ? y$1 : y$1 === void 0 ? y : constant(y$1);
  function line2(data) {
    var i2, n = (data = array(data)).length, d, defined0 = false, buffer;
    if (context == null)
      output = curve(buffer = path());
    for (i2 = 0; i2 <= n; ++i2) {
      if (!(i2 < n && defined(d = data[i2], i2, data)) === defined0) {
        if (defined0 = !defined0)
          output.lineStart();
        else
          output.lineEnd();
      }
      if (defined0)
        output.point(+x$1(d, i2, data), +y$1(d, i2, data));
    }
    if (buffer)
      return output = null, buffer + "" || null;
  }
  line2.x = function(_) {
    return arguments.length ? (x$1 = typeof _ === "function" ? _ : constant(+_), line2) : x$1;
  };
  line2.y = function(_) {
    return arguments.length ? (y$1 = typeof _ === "function" ? _ : constant(+_), line2) : y$1;
  };
  line2.defined = function(_) {
    return arguments.length ? (defined = typeof _ === "function" ? _ : constant(!!_), line2) : defined;
  };
  line2.curve = function(_) {
    return arguments.length ? (curve = _, context != null && (output = curve(context)), line2) : curve;
  };
  line2.context = function(_) {
    return arguments.length ? (_ == null ? context = output = null : output = curve(context = _), line2) : context;
  };
  return line2;
}
function area(x0, y0, y1) {
  var x1 = null, defined = constant(true), context = null, curve = curveLinear, output = null, path = withPath(area2);
  x0 = typeof x0 === "function" ? x0 : x0 === void 0 ? x : constant(+x0);
  y0 = typeof y0 === "function" ? y0 : y0 === void 0 ? constant(0) : constant(+y0);
  y1 = typeof y1 === "function" ? y1 : y1 === void 0 ? y : constant(+y1);
  function area2(data) {
    var i2, j, k, n = (data = array(data)).length, d, defined0 = false, buffer, x0z = new Array(n), y0z = new Array(n);
    if (context == null)
      output = curve(buffer = path());
    for (i2 = 0; i2 <= n; ++i2) {
      if (!(i2 < n && defined(d = data[i2], i2, data)) === defined0) {
        if (defined0 = !defined0) {
          j = i2;
          output.areaStart();
          output.lineStart();
        } else {
          output.lineEnd();
          output.lineStart();
          for (k = i2 - 1; k >= j; --k) {
            output.point(x0z[k], y0z[k]);
          }
          output.lineEnd();
          output.areaEnd();
        }
      }
      if (defined0) {
        x0z[i2] = +x0(d, i2, data), y0z[i2] = +y0(d, i2, data);
        output.point(x1 ? +x1(d, i2, data) : x0z[i2], y1 ? +y1(d, i2, data) : y0z[i2]);
      }
    }
    if (buffer)
      return output = null, buffer + "" || null;
  }
  function arealine() {
    return line$1().defined(defined).curve(curve).context(context);
  }
  area2.x = function(_) {
    return arguments.length ? (x0 = typeof _ === "function" ? _ : constant(+_), x1 = null, area2) : x0;
  };
  area2.x0 = function(_) {
    return arguments.length ? (x0 = typeof _ === "function" ? _ : constant(+_), area2) : x0;
  };
  area2.x1 = function(_) {
    return arguments.length ? (x1 = _ == null ? null : typeof _ === "function" ? _ : constant(+_), area2) : x1;
  };
  area2.y = function(_) {
    return arguments.length ? (y0 = typeof _ === "function" ? _ : constant(+_), y1 = null, area2) : y0;
  };
  area2.y0 = function(_) {
    return arguments.length ? (y0 = typeof _ === "function" ? _ : constant(+_), area2) : y0;
  };
  area2.y1 = function(_) {
    return arguments.length ? (y1 = _ == null ? null : typeof _ === "function" ? _ : constant(+_), area2) : y1;
  };
  area2.lineX0 = area2.lineY0 = function() {
    return arealine().x(x0).y(y0);
  };
  area2.lineY1 = function() {
    return arealine().x(x0).y(y1);
  };
  area2.lineX1 = function() {
    return arealine().x(x1).y(y0);
  };
  area2.defined = function(_) {
    return arguments.length ? (defined = typeof _ === "function" ? _ : constant(!!_), area2) : defined;
  };
  area2.curve = function(_) {
    return arguments.length ? (curve = _, context != null && (output = curve(context)), area2) : curve;
  };
  area2.context = function(_) {
    return arguments.length ? (_ == null ? context = output = null : output = curve(context = _), area2) : context;
  };
  return area2;
}
function point(that, x2, y2) {
  that._context.bezierCurveTo(
    (2 * that._x0 + that._x1) / 3,
    (2 * that._y0 + that._y1) / 3,
    (that._x0 + 2 * that._x1) / 3,
    (that._y0 + 2 * that._y1) / 3,
    (that._x0 + 4 * that._x1 + x2) / 6,
    (that._y0 + 4 * that._y1 + y2) / 6
  );
}
function Basis(context) {
  this._context = context;
}
Basis.prototype = {
  areaStart: function() {
    this._line = 0;
  },
  areaEnd: function() {
    this._line = NaN;
  },
  lineStart: function() {
    this._x0 = this._x1 = this._y0 = this._y1 = NaN;
    this._point = 0;
  },
  lineEnd: function() {
    switch (this._point) {
      case 3:
        point(this, this._x1, this._y1);
      case 2:
        this._context.lineTo(this._x1, this._y1);
        break;
    }
    if (this._line || this._line !== 0 && this._point === 1)
      this._context.closePath();
    this._line = 1 - this._line;
  },
  point: function(x2, y2) {
    x2 = +x2, y2 = +y2;
    switch (this._point) {
      case 0:
        this._point = 1;
        this._line ? this._context.lineTo(x2, y2) : this._context.moveTo(x2, y2);
        break;
      case 1:
        this._point = 2;
        break;
      case 2:
        this._point = 3;
        this._context.lineTo((5 * this._x0 + this._x1) / 6, (5 * this._y0 + this._y1) / 6);
      default:
        point(this, x2, y2);
        break;
    }
    this._x0 = this._x1, this._x1 = x2;
    this._y0 = this._y1, this._y1 = y2;
  }
};
function basis(context) {
  return new Basis(context);
}
function Transform(k, x2, y2) {
  this.k = k;
  this.x = x2;
  this.y = y2;
}
Transform.prototype = {
  constructor: Transform,
  scale: function(k) {
    return k === 1 ? this : new Transform(this.k * k, this.x, this.y);
  },
  translate: function(x2, y2) {
    return x2 === 0 & y2 === 0 ? this : new Transform(this.k, this.x + this.k * x2, this.y + this.k * y2);
  },
  apply: function(point2) {
    return [point2[0] * this.k + this.x, point2[1] * this.k + this.y];
  },
  applyX: function(x2) {
    return x2 * this.k + this.x;
  },
  applyY: function(y2) {
    return y2 * this.k + this.y;
  },
  invert: function(location) {
    return [(location[0] - this.x) / this.k, (location[1] - this.y) / this.k];
  },
  invertX: function(x2) {
    return (x2 - this.x) / this.k;
  },
  invertY: function(y2) {
    return (y2 - this.y) / this.k;
  },
  rescaleX: function(x2) {
    return x2.copy().domain(x2.range().map(this.invertX, this).map(x2.invert, x2));
  },
  rescaleY: function(y2) {
    return y2.copy().domain(y2.range().map(this.invertY, this).map(y2.invert, y2));
  },
  toString: function() {
    return "translate(" + this.x + "," + this.y + ") scale(" + this.k + ")";
  }
};
Transform.prototype;
const kde = (ds, bandwidth, kernelFunction, numPoints) => {
  const [minX, maxX] = extent(ds);
  const spanX = maxX - minX;
  const step = spanX / (numPoints - 1);
  const normalizeX = (x2) => (x2 - minX) / spanX;
  const xValues = Array.from({ length: numPoints }, (_, i2) => minX + i2 * step);
  const kdeValues = xValues.map((x2) => {
    const sum = ds.reduce((prev2, current) => {
      const diffN = (x2 - current) / bandwidth;
      return prev2 + kernelFunction(diffN);
    }, 0);
    return 1 / (ds.length * bandwidth) * sum;
  });
  const domainY = extent(kdeValues);
  const normalizeY = linear().domain(domainY).range([0, 1]);
  return {
    kdeValues: xValues.map((x2, i2) => [normalizeX(x2), normalizeY(kdeValues[i2])]),
    domainY,
    domainX: [0, 1]
  };
};
const kdeEpanechnikov = (k) => (v) => Math.abs(v /= k) <= 1 ? 0.75 * (1 - v * v) / k : 0;
const portal = (element2, mountingPointSelector) => {
  const update2 = async () => {
    if (!document.querySelector(mountingPointSelector)) {
      await tick();
    }
    document.querySelector(mountingPointSelector).appendChild(element2);
  };
  const destroy = async () => {
    if (element2.parentNode) {
      element2.parentNode.removeChild(element2);
    }
  };
  update2();
  return { update: update2, destroy };
};
var has = Object.prototype.hasOwnProperty;
function find(iter, tar, key2) {
  for (key2 of iter.keys()) {
    if (dequal(key2, tar))
      return key2;
  }
}
function dequal(foo, bar) {
  var ctor, len, tmp;
  if (foo === bar)
    return true;
  if (foo && bar && (ctor = foo.constructor) === bar.constructor) {
    if (ctor === Date)
      return foo.getTime() === bar.getTime();
    if (ctor === RegExp)
      return foo.toString() === bar.toString();
    if (ctor === Array) {
      if ((len = foo.length) === bar.length) {
        while (len-- && dequal(foo[len], bar[len]))
          ;
      }
      return len === -1;
    }
    if (ctor === Set) {
      if (foo.size !== bar.size) {
        return false;
      }
      for (len of foo) {
        tmp = len;
        if (tmp && typeof tmp === "object") {
          tmp = find(bar, tmp);
          if (!tmp)
            return false;
        }
        if (!bar.has(tmp))
          return false;
      }
      return true;
    }
    if (ctor === Map) {
      if (foo.size !== bar.size) {
        return false;
      }
      for (len of foo) {
        tmp = len[0];
        if (tmp && typeof tmp === "object") {
          tmp = find(bar, tmp);
          if (!tmp)
            return false;
        }
        if (!dequal(len[1], bar.get(tmp))) {
          return false;
        }
      }
      return true;
    }
    if (ctor === ArrayBuffer) {
      foo = new Uint8Array(foo);
      bar = new Uint8Array(bar);
    } else if (ctor === DataView) {
      if ((len = foo.byteLength) === bar.byteLength) {
        while (len-- && foo.getInt8(len) === bar.getInt8(len))
          ;
      }
      return len === -1;
    }
    if (ArrayBuffer.isView(foo)) {
      if ((len = foo.byteLength) === bar.byteLength) {
        while (len-- && foo[len] === bar[len])
          ;
      }
      return len === -1;
    }
    if (!ctor || typeof foo === "object") {
      len = 0;
      for (ctor in foo) {
        if (has.call(foo, ctor) && ++len && !has.call(bar, ctor))
          return false;
        if (!(ctor in bar) || !dequal(foo[ctor], bar[ctor]))
          return false;
      }
      return Object.keys(bar).length === len;
    }
  }
  return foo !== foo && bar !== bar;
}
function back(array2, index, increment, loop2 = true) {
  const previousIndex = index - increment;
  if (previousIndex <= 0) {
    return loop2 ? array2[array2.length - 1] : array2[0];
  }
  return array2[previousIndex];
}
function forward(array2, index, increment, loop2 = true) {
  const nextIndex = index + increment;
  if (nextIndex > array2.length - 1) {
    return loop2 ? array2[0] : array2[array2.length - 1];
  }
  return array2[nextIndex];
}
function next(array2, index, loop2 = true) {
  if (index === array2.length - 1) {
    return loop2 ? array2[0] : array2[index];
  }
  return array2[index + 1];
}
function prev(array2, currentIndex, loop2 = true) {
  if (currentIndex <= 0) {
    return loop2 ? array2[array2.length - 1] : array2[0];
  }
  return array2[currentIndex - 1];
}
function last(array2) {
  return array2[array2.length - 1];
}
function wrapArray(array2, startIndex) {
  return array2.map((_, index) => array2[(startIndex + index) % array2.length]);
}
function toggle(item, array2, compare2 = dequal) {
  const itemIdx = array2.findIndex((innerItem) => compare2(innerItem, item));
  if (itemIdx !== -1) {
    array2.splice(itemIdx, 1);
  } else {
    array2.push(item);
  }
  return array2;
}
function styleToString(style2) {
  return Object.keys(style2).reduce((str, key2) => {
    if (style2[key2] === void 0)
      return str;
    return str + `${key2}:${style2[key2]};`;
  }, "");
}
function disabledAttr(disabled) {
  return disabled ? true : void 0;
}
const hiddenInputAttrs = {
  type: "hidden",
  "aria-hidden": true,
  hidden: true,
  tabIndex: -1,
  style: styleToString({
    position: "absolute",
    opacity: 0,
    "pointer-events": "none",
    margin: 0,
    transform: "translateX(-100%)"
  })
};
function lightable(value) {
  function subscribe2(run2) {
    run2(value);
    return () => {
    };
  }
  return { subscribe: subscribe2 };
}
function getElementByMeltId(id2) {
  if (!isBrowser)
    return null;
  const el = document.querySelector(`[data-melt-id="${id2}"]`);
  return isHTMLElement$1(el) ? el : null;
}
const hiddenAction = (obj) => {
  return new Proxy(obj, {
    get(target, prop, receiver) {
      return Reflect.get(target, prop, receiver);
    },
    ownKeys(target) {
      return Reflect.ownKeys(target).filter((key2) => key2 !== "action");
    }
  });
};
const isFunctionWithParams = (fn) => {
  return typeof fn === "function";
};
function builder(name2, args) {
  const { stores, action, returned } = args ?? {};
  const derivedStore = (() => {
    if (stores && returned) {
      return derived(stores, (values) => {
        const result = returned(values);
        if (isFunctionWithParams(result)) {
          const fn = (...args2) => {
            return hiddenAction({
              ...result(...args2),
              [`data-melt-${name2}`]: "",
              action: action ?? noop
            });
          };
          fn.action = action ?? noop;
          return fn;
        }
        return hiddenAction({
          ...result,
          [`data-melt-${name2}`]: "",
          action: action ?? noop
        });
      });
    } else {
      const returnedFn = returned;
      const result = returnedFn == null ? void 0 : returnedFn();
      if (isFunctionWithParams(result)) {
        const resultFn = (...args2) => {
          return hiddenAction({
            ...result(...args2),
            [`data-melt-${name2}`]: "",
            action: action ?? noop
          });
        };
        resultFn.action = action ?? noop;
        return lightable(resultFn);
      }
      return lightable(hiddenAction({
        ...result,
        [`data-melt-${name2}`]: "",
        action: action ?? noop
      }));
    }
  })();
  const actionFn = action ?? (() => {
  });
  actionFn.subscribe = derivedStore.subscribe;
  return actionFn;
}
function builderArray(name2, args) {
  const { stores, returned, action } = args;
  const { subscribe: subscribe2 } = derived(stores, (values) => returned(values).map((value) => hiddenAction({
    ...value,
    [`data-melt-${name2}`]: "",
    action: action ?? noop
  })));
  const actionFn = action ?? (() => {
  });
  actionFn.subscribe = subscribe2;
  return actionFn;
}
function createElHelpers(prefix) {
  const name2 = (part) => part ? `${prefix}-${part}` : prefix;
  const attribute = (part) => `data-melt-${prefix}${part ? `-${part}` : ""}`;
  const selector2 = (part) => `[data-melt-${prefix}${part ? `-${part}` : ""}]`;
  const getEl = (part) => document.querySelector(selector2(part));
  return {
    name: name2,
    attribute,
    selector: selector2,
    getEl
  };
}
const isBrowser = typeof document !== "undefined";
const isFunction = (v) => typeof v === "function";
function isElement$1(element2) {
  return element2 instanceof Element;
}
function isHTMLElement$1(element2) {
  return element2 instanceof HTMLElement;
}
function isHTMLInputElement(element2) {
  return element2 instanceof HTMLInputElement;
}
function isHTMLLabelElement(element2) {
  return element2 instanceof HTMLLabelElement;
}
function isHTMLButtonElement(element2) {
  return element2 instanceof HTMLButtonElement;
}
function isElementDisabled(element2) {
  const ariaDisabled = element2.getAttribute("aria-disabled");
  const disabled = element2.getAttribute("disabled");
  const dataDisabled = element2.hasAttribute("data-disabled");
  if (ariaDisabled === "true" || disabled !== null || dataDisabled) {
    return true;
  }
  return false;
}
function isContentEditable$1(element2) {
  if (!isHTMLElement$1(element2))
    return false;
  return element2.isContentEditable;
}
function isObject(value) {
  return value !== null && typeof value === "object";
}
function isReadable(value) {
  return isObject(value) && "subscribe" in value;
}
function executeCallbacks(...callbacks) {
  return (...args) => {
    for (const callback of callbacks) {
      if (typeof callback === "function") {
        callback(...args);
      }
    }
  };
}
function noop() {
}
function addEventListener(target, event, handler, options) {
  const events = Array.isArray(event) ? event : [event];
  events.forEach((_event) => target.addEventListener(_event, handler, options));
  return () => {
    events.forEach((_event) => target.removeEventListener(_event, handler, options));
  };
}
function addMeltEventListener(target, event, handler, options) {
  const events = Array.isArray(event) ? event : [event];
  if (typeof handler === "function") {
    const handlerWithMelt = withMelt((_event) => handler(_event));
    events.forEach((_event) => target.addEventListener(_event, handlerWithMelt, options));
    return () => {
      events.forEach((_event) => target.removeEventListener(_event, handlerWithMelt, options));
    };
  }
  return () => noop();
}
function dispatchMeltEvent(originalEvent) {
  const node = originalEvent.currentTarget;
  if (!isHTMLElement$1(node))
    return null;
  const customMeltEvent = new CustomEvent(`m-${originalEvent.type}`, {
    detail: {
      originalEvent
    },
    cancelable: true
  });
  node.dispatchEvent(customMeltEvent);
  return customMeltEvent;
}
function withMelt(handler) {
  return (event) => {
    const customEvent = dispatchMeltEvent(event);
    if (customEvent == null ? void 0 : customEvent.defaultPrevented)
      return;
    return handler(event);
  };
}
function addHighlight(element2) {
  element2.setAttribute("data-highlighted", "");
}
function removeHighlight(element2) {
  element2.removeAttribute("data-highlighted");
}
function getOptions(el) {
  return Array.from(el.querySelectorAll('[role="option"]:not([data-disabled])')).filter((el2) => isHTMLElement$1(el2));
}
function omit(obj, ...keys) {
  const result = {};
  for (const key2 of Object.keys(obj)) {
    if (!keys.includes(key2)) {
      result[key2] = obj[key2];
    }
  }
  return result;
}
function stripValues(inputObject, toStrip, recursive) {
  return Object.fromEntries(Object.entries(inputObject).filter(([_, value]) => !dequal(value, toStrip)));
}
const overridable = (store2, onChange) => {
  const update2 = (updater, sideEffect) => {
    store2.update((curr) => {
      const next2 = updater(curr);
      let res = next2;
      if (onChange) {
        res = onChange({ curr, next: next2 });
      }
      sideEffect == null ? void 0 : sideEffect(res);
      return res;
    });
  };
  const set2 = (curr) => {
    update2(() => curr);
  };
  return {
    ...store2,
    update: update2,
    set: set2
  };
};
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
let urlAlphabet = "useandom-26T198340PX75pxJACKVERYMINDBUSHWOLF_GQZbfghjklqvwyzrict";
let nanoid = (size2 = 21) => {
  let id2 = "";
  let i2 = size2;
  while (i2--) {
    id2 += urlAlphabet[Math.random() * 64 | 0];
  }
  return id2;
};
function generateId() {
  return nanoid(10);
}
function generateIds(args) {
  return args.reduce((acc, curr) => {
    acc[curr] = generateId();
    return acc;
  }, {});
}
const kbd = {
  ALT: "Alt",
  ARROW_DOWN: "ArrowDown",
  ARROW_LEFT: "ArrowLeft",
  ARROW_RIGHT: "ArrowRight",
  ARROW_UP: "ArrowUp",
  BACKSPACE: "Backspace",
  CAPS_LOCK: "CapsLock",
  CONTROL: "Control",
  DELETE: "Delete",
  END: "End",
  ENTER: "Enter",
  ESCAPE: "Escape",
  F1: "F1",
  F10: "F10",
  F11: "F11",
  F12: "F12",
  F2: "F2",
  F3: "F3",
  F4: "F4",
  F5: "F5",
  F6: "F6",
  F7: "F7",
  F8: "F8",
  F9: "F9",
  HOME: "Home",
  META: "Meta",
  PAGE_DOWN: "PageDown",
  PAGE_UP: "PageUp",
  SHIFT: "Shift",
  SPACE: " ",
  TAB: "Tab",
  CTRL: "Control",
  ASTERISK: "*",
  A: "a",
  P: "p"
};
const FIRST_KEYS = [kbd.ARROW_DOWN, kbd.PAGE_UP, kbd.HOME];
const LAST_KEYS = [kbd.ARROW_UP, kbd.PAGE_DOWN, kbd.END];
const FIRST_LAST_KEYS = [...FIRST_KEYS, ...LAST_KEYS];
function debounce(fn, wait = 500) {
  let timeout2 = null;
  return function(...args) {
    const later = () => {
      timeout2 = null;
      fn(...args);
    };
    timeout2 && clearTimeout(timeout2);
    timeout2 = setTimeout(later, wait);
  };
}
const isDom = () => typeof window !== "undefined";
function getPlatform() {
  const agent = navigator.userAgentData;
  return (agent == null ? void 0 : agent.platform) ?? navigator.platform;
}
const pt = (v) => isDom() && v.test(getPlatform().toLowerCase());
const isTouchDevice = () => isDom() && !!navigator.maxTouchPoints;
const isMac = () => pt(/^mac/) && !isTouchDevice();
const isApple = () => pt(/mac|iphone|ipad|ipod/i);
const isIos = () => isApple() && !isMac();
const LOCK_CLASSNAME = "data-melt-scroll-lock";
function assignStyle(el, style2) {
  if (!el)
    return;
  const previousStyle = el.style.cssText;
  Object.assign(el.style, style2);
  return () => {
    el.style.cssText = previousStyle;
  };
}
function setCSSProperty(el, property, value) {
  if (!el)
    return;
  const previousValue = el.style.getPropertyValue(property);
  el.style.setProperty(property, value);
  return () => {
    if (previousValue) {
      el.style.setProperty(property, previousValue);
    } else {
      el.style.removeProperty(property);
    }
  };
}
function getPaddingProperty(documentElement) {
  const documentLeft = documentElement.getBoundingClientRect().left;
  const scrollbarX = Math.round(documentLeft) + documentElement.scrollLeft;
  return scrollbarX ? "paddingLeft" : "paddingRight";
}
function removeScroll(_document) {
  const doc = _document ?? document;
  const win = doc.defaultView ?? window;
  const { documentElement, body } = doc;
  const locked = body.hasAttribute(LOCK_CLASSNAME);
  if (locked)
    return noop;
  body.setAttribute(LOCK_CLASSNAME, "");
  const scrollbarWidth = win.innerWidth - documentElement.clientWidth;
  const setScrollbarWidthProperty = () => setCSSProperty(documentElement, "--scrollbar-width", `${scrollbarWidth}px`);
  const paddingProperty = getPaddingProperty(documentElement);
  const scrollbarSidePadding = win.getComputedStyle(body)[paddingProperty];
  const setStyle = () => assignStyle(body, {
    overflow: "hidden",
    [paddingProperty]: `calc(${scrollbarSidePadding} + ${scrollbarWidth}px)`
  });
  const setIOSStyle = () => {
    const { scrollX, scrollY, visualViewport } = win;
    const offsetLeft = (visualViewport == null ? void 0 : visualViewport.offsetLeft) ?? 0;
    const offsetTop = (visualViewport == null ? void 0 : visualViewport.offsetTop) ?? 0;
    const restoreStyle = assignStyle(body, {
      position: "fixed",
      overflow: "hidden",
      top: `${-(scrollY - Math.floor(offsetTop))}px`,
      left: `${-(scrollX - Math.floor(offsetLeft))}px`,
      right: "0",
      [paddingProperty]: `calc(${scrollbarSidePadding} + ${scrollbarWidth}px)`
    });
    return () => {
      restoreStyle == null ? void 0 : restoreStyle();
      win.scrollTo(scrollX, scrollY);
    };
  };
  const cleanups = [setScrollbarWidthProperty(), isIos() ? setIOSStyle() : setStyle()];
  return () => {
    cleanups.forEach((fn) => fn == null ? void 0 : fn());
    body.removeAttribute(LOCK_CLASSNAME);
  };
}
function derivedVisible(obj) {
  const { open, forceVisible, activeTrigger } = obj;
  return derived([open, forceVisible, activeTrigger], ([$open, $forceVisible, $activeTrigger]) => ($open || $forceVisible) && $activeTrigger !== null);
}
const safeOnMount = (fn) => {
  try {
    onMount(fn);
  } catch {
    return fn();
  }
};
const safeOnDestroy = (fn) => {
  try {
    onDestroy(fn);
  } catch {
    return fn();
  }
};
function derivedWithUnsubscribe(stores, fn) {
  let unsubscribers = [];
  const onUnsubscribe = (cb) => {
    unsubscribers.push(cb);
  };
  const unsubscribe2 = () => {
    unsubscribers.forEach((fn2) => fn2());
    unsubscribers = [];
  };
  const derivedStore = derived(stores, ($storeValues) => {
    unsubscribe2();
    return fn($storeValues, onUnsubscribe);
  });
  safeOnDestroy(unsubscribe2);
  const subscribe2 = (...args) => {
    const unsub = derivedStore.subscribe(...args);
    return () => {
      unsub();
      unsubscribe2();
    };
  };
  return {
    ...derivedStore,
    subscribe: subscribe2
  };
}
function effect(stores, fn) {
  const unsub = derivedWithUnsubscribe(stores, (stores2, onUnsubscribe) => {
    return {
      stores: stores2,
      onUnsubscribe
    };
  }).subscribe(({ stores: stores2, onUnsubscribe }) => {
    const returned = fn(stores2);
    if (returned) {
      onUnsubscribe(returned);
    }
  });
  safeOnDestroy(unsub);
  return unsub;
}
function toWritableStores(properties) {
  const result = {};
  Object.keys(properties).forEach((key2) => {
    const propertyKey = key2;
    const value = properties[propertyKey];
    result[propertyKey] = writable(value);
  });
  return result;
}
function handleRovingFocus(nextElement) {
  if (!isBrowser)
    return;
  sleep(1).then(() => {
    const currentFocusedElement = document.activeElement;
    if (!isHTMLElement$1(currentFocusedElement) || currentFocusedElement === nextElement)
      return;
    currentFocusedElement.tabIndex = -1;
    if (nextElement) {
      nextElement.tabIndex = 0;
      nextElement.focus();
    }
  });
}
const ignoredKeys = /* @__PURE__ */ new Set(["Shift", "Control", "Alt", "Meta", "CapsLock", "NumLock"]);
const defaults$2 = {
  onMatch: handleRovingFocus,
  getCurrentItem: () => document.activeElement
};
function createTypeaheadSearch(args = {}) {
  const withDefaults = { ...defaults$2, ...args };
  const typed = writable([]);
  const resetTyped = debounce(() => {
    typed.update(() => []);
  });
  const handleTypeaheadSearch = (key2, items) => {
    if (ignoredKeys.has(key2))
      return;
    const currentItem = withDefaults.getCurrentItem();
    const $typed = get_store_value(typed);
    if (!Array.isArray($typed)) {
      return;
    }
    $typed.push(key2.toLowerCase());
    typed.set($typed);
    const candidateItems = items.filter((item) => {
      if (item.getAttribute("disabled") === "true" || item.getAttribute("aria-disabled") === "true" || item.hasAttribute("data-disabled")) {
        return false;
      }
      return true;
    });
    const isRepeated = $typed.length > 1 && $typed.every((char) => char === $typed[0]);
    const normalizeSearch = isRepeated ? $typed[0] : $typed.join("");
    const currentItemIndex = isHTMLElement$1(currentItem) ? candidateItems.indexOf(currentItem) : -1;
    let wrappedItems = wrapArray(candidateItems, Math.max(currentItemIndex, 0));
    const excludeCurrentItem = normalizeSearch.length === 1;
    if (excludeCurrentItem) {
      wrappedItems = wrappedItems.filter((v) => v !== currentItem);
    }
    const nextItem = wrappedItems.find((item) => (item == null ? void 0 : item.innerText) && item.innerText.toLowerCase().startsWith(normalizeSearch.toLowerCase()));
    if (isHTMLElement$1(nextItem) && nextItem !== currentItem) {
      withDefaults.onMatch(nextItem);
    }
    resetTyped();
  };
  return {
    typed,
    resetTyped,
    handleTypeaheadSearch
  };
}
function getPortalParent(node) {
  let parent = node.parentElement;
  while (isHTMLElement$1(parent) && !parent.hasAttribute("data-portal")) {
    parent = parent.parentElement;
  }
  return parent || "body";
}
function getPortalDestination(node, portalProp) {
  const portalParent = getPortalParent(node);
  if (portalProp !== void 0)
    return portalProp;
  if (portalParent === "body")
    return document.body;
  return null;
}
function createClickOutsideIgnore(meltId) {
  return (e) => {
    const target = e.target;
    const triggerEl = getElementByMeltId(meltId);
    if (!triggerEl || !isElement$1(target))
      return false;
    const id2 = triggerEl.id;
    if (isHTMLLabelElement(target) && id2 === target.htmlFor) {
      return true;
    }
    if (target.closest(`label[for="${id2}"]`)) {
      return true;
    }
    return false;
  };
}
function snapValueToStep(value, min2, max2, step) {
  const remainder = (value - (isNaN(min2) ? 0 : min2)) % step;
  let snappedValue = Math.abs(remainder) * 2 >= step ? value + Math.sign(remainder) * (step - Math.abs(remainder)) : value - remainder;
  if (!isNaN(min2)) {
    if (snappedValue < min2) {
      snappedValue = min2;
    } else if (!isNaN(max2) && snappedValue > max2) {
      snappedValue = min2 + Math.floor((max2 - min2) / step) * step;
    }
  } else if (!isNaN(max2) && snappedValue > max2) {
    snappedValue = Math.floor(max2 / step) * step;
  }
  const string = step.toString();
  const index = string.indexOf(".");
  const precision = index >= 0 ? string.length - index : 0;
  if (precision > 0) {
    const pow = Math.pow(10, precision);
    snappedValue = Math.round(snappedValue * pow) / pow;
  }
  return snappedValue;
}
const documentClickStore = readable(void 0, (set2) => {
  function clicked(event) {
    set2(event);
    set2(void 0);
  }
  const unsubscribe2 = addEventListener(document, "pointerup", clicked, {
    passive: false,
    capture: true
  });
  return unsubscribe2;
});
const useClickOutside = (node, config = {}) => {
  let options = { enabled: true, ...config };
  function isEnabled() {
    return typeof options.enabled === "boolean" ? options.enabled : get_store_value(options.enabled);
  }
  const unsubscribe2 = documentClickStore.subscribe((e) => {
    var _a;
    if (!isEnabled() || !e || e.target === node) {
      return;
    }
    const composedPath = e.composedPath();
    if (composedPath.includes(node))
      return;
    if (options.ignore) {
      if (isFunction(options.ignore)) {
        if (options.ignore(e))
          return;
      } else if (Array.isArray(options.ignore)) {
        if (options.ignore.length > 0 && options.ignore.some((ignoreEl) => {
          return ignoreEl && (e.target === ignoreEl || composedPath.includes(ignoreEl));
        }))
          return;
      }
    }
    (_a = options.handler) == null ? void 0 : _a.call(options, e);
  });
  return {
    update(params) {
      options = { ...options, ...params };
    },
    destroy() {
      unsubscribe2();
    }
  };
};
const documentEscapeKeyStore = readable(void 0, (set2) => {
  function keydown(event) {
    if (event && event.key === kbd.ESCAPE) {
      set2(event);
    }
    set2(void 0);
  }
  const unsubscribe2 = addEventListener(document, "keydown", keydown, {
    passive: false
  });
  return unsubscribe2;
});
const useEscapeKeydown = (node, config = {}) => {
  let unsub = noop;
  function update2(config2 = {}) {
    unsub();
    const options = { enabled: true, ...config2 };
    const enabled = isReadable(options.enabled) ? options.enabled : readable(options.enabled);
    unsub = executeCallbacks(
      // Handle escape keydowns
      documentEscapeKeyStore.subscribe((e) => {
        var _a;
        if (!e || !get_store_value(enabled))
          return;
        const target = e.target;
        if (!isHTMLElement$1(target) || target.closest("[data-escapee]") !== node) {
          return;
        }
        e.preventDefault();
        if (options.ignore) {
          if (isFunction(options.ignore)) {
            if (options.ignore(e))
              return;
          } else if (Array.isArray(options.ignore)) {
            if (options.ignore.length > 0 && options.ignore.some((ignoreEl) => {
              return ignoreEl && target === ignoreEl;
            }))
              return;
          }
        }
        (_a = options.handler) == null ? void 0 : _a.call(options, e);
      }),
      effect(enabled, ($enabled) => {
        if ($enabled) {
          node.dataset.escapee = "";
        } else {
          delete node.dataset.escapee;
        }
      })
    );
  }
  update2(config);
  return {
    update: update2,
    destroy() {
      node.removeAttribute("data-escapee");
      unsub();
    }
  };
};
const min = Math.min;
const max = Math.max;
const round = Math.round;
const floor = Math.floor;
const createCoords = (v) => ({
  x: v,
  y: v
});
const oppositeSideMap = {
  left: "right",
  right: "left",
  bottom: "top",
  top: "bottom"
};
const oppositeAlignmentMap = {
  start: "end",
  end: "start"
};
function clamp(start2, value, end) {
  return max(start2, min(value, end));
}
function evaluate(value, param) {
  return typeof value === "function" ? value(param) : value;
}
function getSide(placement) {
  return placement.split("-")[0];
}
function getAlignment(placement) {
  return placement.split("-")[1];
}
function getOppositeAxis(axis2) {
  return axis2 === "x" ? "y" : "x";
}
function getAxisLength(axis2) {
  return axis2 === "y" ? "height" : "width";
}
function getSideAxis(placement) {
  return ["top", "bottom"].includes(getSide(placement)) ? "y" : "x";
}
function getAlignmentAxis(placement) {
  return getOppositeAxis(getSideAxis(placement));
}
function getAlignmentSides(placement, rects, rtl) {
  if (rtl === void 0) {
    rtl = false;
  }
  const alignment = getAlignment(placement);
  const alignmentAxis = getAlignmentAxis(placement);
  const length = getAxisLength(alignmentAxis);
  let mainAlignmentSide = alignmentAxis === "x" ? alignment === (rtl ? "end" : "start") ? "right" : "left" : alignment === "start" ? "bottom" : "top";
  if (rects.reference[length] > rects.floating[length]) {
    mainAlignmentSide = getOppositePlacement(mainAlignmentSide);
  }
  return [mainAlignmentSide, getOppositePlacement(mainAlignmentSide)];
}
function getExpandedPlacements(placement) {
  const oppositePlacement = getOppositePlacement(placement);
  return [getOppositeAlignmentPlacement(placement), oppositePlacement, getOppositeAlignmentPlacement(oppositePlacement)];
}
function getOppositeAlignmentPlacement(placement) {
  return placement.replace(/start|end/g, (alignment) => oppositeAlignmentMap[alignment]);
}
function getSideList(side, isStart, rtl) {
  const lr = ["left", "right"];
  const rl = ["right", "left"];
  const tb = ["top", "bottom"];
  const bt = ["bottom", "top"];
  switch (side) {
    case "top":
    case "bottom":
      if (rtl)
        return isStart ? rl : lr;
      return isStart ? lr : rl;
    case "left":
    case "right":
      return isStart ? tb : bt;
    default:
      return [];
  }
}
function getOppositeAxisPlacements(placement, flipAlignment, direction, rtl) {
  const alignment = getAlignment(placement);
  let list2 = getSideList(getSide(placement), direction === "start", rtl);
  if (alignment) {
    list2 = list2.map((side) => side + "-" + alignment);
    if (flipAlignment) {
      list2 = list2.concat(list2.map(getOppositeAlignmentPlacement));
    }
  }
  return list2;
}
function getOppositePlacement(placement) {
  return placement.replace(/left|right|bottom|top/g, (side) => oppositeSideMap[side]);
}
function expandPaddingObject(padding) {
  return {
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    ...padding
  };
}
function getPaddingObject(padding) {
  return typeof padding !== "number" ? expandPaddingObject(padding) : {
    top: padding,
    right: padding,
    bottom: padding,
    left: padding
  };
}
function rectToClientRect(rect) {
  return {
    ...rect,
    top: rect.y,
    left: rect.x,
    right: rect.x + rect.width,
    bottom: rect.y + rect.height
  };
}
function computeCoordsFromPlacement(_ref, placement, rtl) {
  let {
    reference,
    floating
  } = _ref;
  const sideAxis = getSideAxis(placement);
  const alignmentAxis = getAlignmentAxis(placement);
  const alignLength = getAxisLength(alignmentAxis);
  const side = getSide(placement);
  const isVertical = sideAxis === "y";
  const commonX = reference.x + reference.width / 2 - floating.width / 2;
  const commonY = reference.y + reference.height / 2 - floating.height / 2;
  const commonAlign = reference[alignLength] / 2 - floating[alignLength] / 2;
  let coords;
  switch (side) {
    case "top":
      coords = {
        x: commonX,
        y: reference.y - floating.height
      };
      break;
    case "bottom":
      coords = {
        x: commonX,
        y: reference.y + reference.height
      };
      break;
    case "right":
      coords = {
        x: reference.x + reference.width,
        y: commonY
      };
      break;
    case "left":
      coords = {
        x: reference.x - floating.width,
        y: commonY
      };
      break;
    default:
      coords = {
        x: reference.x,
        y: reference.y
      };
  }
  switch (getAlignment(placement)) {
    case "start":
      coords[alignmentAxis] -= commonAlign * (rtl && isVertical ? -1 : 1);
      break;
    case "end":
      coords[alignmentAxis] += commonAlign * (rtl && isVertical ? -1 : 1);
      break;
  }
  return coords;
}
const computePosition$1 = async (reference, floating, config) => {
  const {
    placement = "bottom",
    strategy = "absolute",
    middleware = [],
    platform: platform2
  } = config;
  const validMiddleware = middleware.filter(Boolean);
  const rtl = await (platform2.isRTL == null ? void 0 : platform2.isRTL(floating));
  let rects = await platform2.getElementRects({
    reference,
    floating,
    strategy
  });
  let {
    x: x2,
    y: y2
  } = computeCoordsFromPlacement(rects, placement, rtl);
  let statefulPlacement = placement;
  let middlewareData = {};
  let resetCount = 0;
  for (let i2 = 0; i2 < validMiddleware.length; i2++) {
    const {
      name: name2,
      fn
    } = validMiddleware[i2];
    const {
      x: nextX,
      y: nextY,
      data,
      reset
    } = await fn({
      x: x2,
      y: y2,
      initialPlacement: placement,
      placement: statefulPlacement,
      strategy,
      middlewareData,
      rects,
      platform: platform2,
      elements: {
        reference,
        floating
      }
    });
    x2 = nextX != null ? nextX : x2;
    y2 = nextY != null ? nextY : y2;
    middlewareData = {
      ...middlewareData,
      [name2]: {
        ...middlewareData[name2],
        ...data
      }
    };
    if (reset && resetCount <= 50) {
      resetCount++;
      if (typeof reset === "object") {
        if (reset.placement) {
          statefulPlacement = reset.placement;
        }
        if (reset.rects) {
          rects = reset.rects === true ? await platform2.getElementRects({
            reference,
            floating,
            strategy
          }) : reset.rects;
        }
        ({
          x: x2,
          y: y2
        } = computeCoordsFromPlacement(rects, statefulPlacement, rtl));
      }
      i2 = -1;
      continue;
    }
  }
  return {
    x: x2,
    y: y2,
    placement: statefulPlacement,
    strategy,
    middlewareData
  };
};
async function detectOverflow(state, options) {
  var _await$platform$isEle;
  if (options === void 0) {
    options = {};
  }
  const {
    x: x2,
    y: y2,
    platform: platform2,
    rects,
    elements,
    strategy
  } = state;
  const {
    boundary = "clippingAncestors",
    rootBoundary = "viewport",
    elementContext = "floating",
    altBoundary = false,
    padding = 0
  } = evaluate(options, state);
  const paddingObject = getPaddingObject(padding);
  const altContext = elementContext === "floating" ? "reference" : "floating";
  const element2 = elements[altBoundary ? altContext : elementContext];
  const clippingClientRect = rectToClientRect(await platform2.getClippingRect({
    element: ((_await$platform$isEle = await (platform2.isElement == null ? void 0 : platform2.isElement(element2))) != null ? _await$platform$isEle : true) ? element2 : element2.contextElement || await (platform2.getDocumentElement == null ? void 0 : platform2.getDocumentElement(elements.floating)),
    boundary,
    rootBoundary,
    strategy
  }));
  const rect = elementContext === "floating" ? {
    ...rects.floating,
    x: x2,
    y: y2
  } : rects.reference;
  const offsetParent = await (platform2.getOffsetParent == null ? void 0 : platform2.getOffsetParent(elements.floating));
  const offsetScale = await (platform2.isElement == null ? void 0 : platform2.isElement(offsetParent)) ? await (platform2.getScale == null ? void 0 : platform2.getScale(offsetParent)) || {
    x: 1,
    y: 1
  } : {
    x: 1,
    y: 1
  };
  const elementClientRect = rectToClientRect(platform2.convertOffsetParentRelativeRectToViewportRelativeRect ? await platform2.convertOffsetParentRelativeRectToViewportRelativeRect({
    rect,
    offsetParent,
    strategy
  }) : rect);
  return {
    top: (clippingClientRect.top - elementClientRect.top + paddingObject.top) / offsetScale.y,
    bottom: (elementClientRect.bottom - clippingClientRect.bottom + paddingObject.bottom) / offsetScale.y,
    left: (clippingClientRect.left - elementClientRect.left + paddingObject.left) / offsetScale.x,
    right: (elementClientRect.right - clippingClientRect.right + paddingObject.right) / offsetScale.x
  };
}
const arrow$1 = (options) => ({
  name: "arrow",
  options,
  async fn(state) {
    const {
      x: x2,
      y: y2,
      placement,
      rects,
      platform: platform2,
      elements,
      middlewareData
    } = state;
    const {
      element: element2,
      padding = 0
    } = evaluate(options, state) || {};
    if (element2 == null) {
      return {};
    }
    const paddingObject = getPaddingObject(padding);
    const coords = {
      x: x2,
      y: y2
    };
    const axis2 = getAlignmentAxis(placement);
    const length = getAxisLength(axis2);
    const arrowDimensions = await platform2.getDimensions(element2);
    const isYAxis = axis2 === "y";
    const minProp = isYAxis ? "top" : "left";
    const maxProp = isYAxis ? "bottom" : "right";
    const clientProp = isYAxis ? "clientHeight" : "clientWidth";
    const endDiff = rects.reference[length] + rects.reference[axis2] - coords[axis2] - rects.floating[length];
    const startDiff = coords[axis2] - rects.reference[axis2];
    const arrowOffsetParent = await (platform2.getOffsetParent == null ? void 0 : platform2.getOffsetParent(element2));
    let clientSize = arrowOffsetParent ? arrowOffsetParent[clientProp] : 0;
    if (!clientSize || !await (platform2.isElement == null ? void 0 : platform2.isElement(arrowOffsetParent))) {
      clientSize = elements.floating[clientProp] || rects.floating[length];
    }
    const centerToReference = endDiff / 2 - startDiff / 2;
    const largestPossiblePadding = clientSize / 2 - arrowDimensions[length] / 2 - 1;
    const minPadding = min(paddingObject[minProp], largestPossiblePadding);
    const maxPadding = min(paddingObject[maxProp], largestPossiblePadding);
    const min$1 = minPadding;
    const max2 = clientSize - arrowDimensions[length] - maxPadding;
    const center2 = clientSize / 2 - arrowDimensions[length] / 2 + centerToReference;
    const offset2 = clamp(min$1, center2, max2);
    const shouldAddOffset = !middlewareData.arrow && getAlignment(placement) != null && center2 != offset2 && rects.reference[length] / 2 - (center2 < min$1 ? minPadding : maxPadding) - arrowDimensions[length] / 2 < 0;
    const alignmentOffset = shouldAddOffset ? center2 < min$1 ? center2 - min$1 : center2 - max2 : 0;
    return {
      [axis2]: coords[axis2] + alignmentOffset,
      data: {
        [axis2]: offset2,
        centerOffset: center2 - offset2 - alignmentOffset,
        ...shouldAddOffset && {
          alignmentOffset
        }
      },
      reset: shouldAddOffset
    };
  }
});
const flip$2 = function(options) {
  if (options === void 0) {
    options = {};
  }
  return {
    name: "flip",
    options,
    async fn(state) {
      var _middlewareData$arrow, _middlewareData$flip;
      const {
        placement,
        middlewareData,
        rects,
        initialPlacement,
        platform: platform2,
        elements
      } = state;
      const {
        mainAxis: checkMainAxis = true,
        crossAxis: checkCrossAxis = true,
        fallbackPlacements: specifiedFallbackPlacements,
        fallbackStrategy = "bestFit",
        fallbackAxisSideDirection = "none",
        flipAlignment = true,
        ...detectOverflowOptions
      } = evaluate(options, state);
      if ((_middlewareData$arrow = middlewareData.arrow) != null && _middlewareData$arrow.alignmentOffset) {
        return {};
      }
      const side = getSide(placement);
      const isBasePlacement = getSide(initialPlacement) === initialPlacement;
      const rtl = await (platform2.isRTL == null ? void 0 : platform2.isRTL(elements.floating));
      const fallbackPlacements = specifiedFallbackPlacements || (isBasePlacement || !flipAlignment ? [getOppositePlacement(initialPlacement)] : getExpandedPlacements(initialPlacement));
      if (!specifiedFallbackPlacements && fallbackAxisSideDirection !== "none") {
        fallbackPlacements.push(...getOppositeAxisPlacements(initialPlacement, flipAlignment, fallbackAxisSideDirection, rtl));
      }
      const placements = [initialPlacement, ...fallbackPlacements];
      const overflow = await detectOverflow(state, detectOverflowOptions);
      const overflows = [];
      let overflowsData = ((_middlewareData$flip = middlewareData.flip) == null ? void 0 : _middlewareData$flip.overflows) || [];
      if (checkMainAxis) {
        overflows.push(overflow[side]);
      }
      if (checkCrossAxis) {
        const sides = getAlignmentSides(placement, rects, rtl);
        overflows.push(overflow[sides[0]], overflow[sides[1]]);
      }
      overflowsData = [...overflowsData, {
        placement,
        overflows
      }];
      if (!overflows.every((side2) => side2 <= 0)) {
        var _middlewareData$flip2, _overflowsData$filter;
        const nextIndex = (((_middlewareData$flip2 = middlewareData.flip) == null ? void 0 : _middlewareData$flip2.index) || 0) + 1;
        const nextPlacement = placements[nextIndex];
        if (nextPlacement) {
          return {
            data: {
              index: nextIndex,
              overflows: overflowsData
            },
            reset: {
              placement: nextPlacement
            }
          };
        }
        let resetPlacement = (_overflowsData$filter = overflowsData.filter((d) => d.overflows[0] <= 0).sort((a, b) => a.overflows[1] - b.overflows[1])[0]) == null ? void 0 : _overflowsData$filter.placement;
        if (!resetPlacement) {
          switch (fallbackStrategy) {
            case "bestFit": {
              var _overflowsData$map$so;
              const placement2 = (_overflowsData$map$so = overflowsData.map((d) => [d.placement, d.overflows.filter((overflow2) => overflow2 > 0).reduce((acc, overflow2) => acc + overflow2, 0)]).sort((a, b) => a[1] - b[1])[0]) == null ? void 0 : _overflowsData$map$so[0];
              if (placement2) {
                resetPlacement = placement2;
              }
              break;
            }
            case "initialPlacement":
              resetPlacement = initialPlacement;
              break;
          }
        }
        if (placement !== resetPlacement) {
          return {
            reset: {
              placement: resetPlacement
            }
          };
        }
      }
      return {};
    }
  };
};
async function convertValueToCoords(state, options) {
  const {
    placement,
    platform: platform2,
    elements
  } = state;
  const rtl = await (platform2.isRTL == null ? void 0 : platform2.isRTL(elements.floating));
  const side = getSide(placement);
  const alignment = getAlignment(placement);
  const isVertical = getSideAxis(placement) === "y";
  const mainAxisMulti = ["left", "top"].includes(side) ? -1 : 1;
  const crossAxisMulti = rtl && isVertical ? -1 : 1;
  const rawValue = evaluate(options, state);
  let {
    mainAxis,
    crossAxis,
    alignmentAxis
  } = typeof rawValue === "number" ? {
    mainAxis: rawValue,
    crossAxis: 0,
    alignmentAxis: null
  } : {
    mainAxis: 0,
    crossAxis: 0,
    alignmentAxis: null,
    ...rawValue
  };
  if (alignment && typeof alignmentAxis === "number") {
    crossAxis = alignment === "end" ? alignmentAxis * -1 : alignmentAxis;
  }
  return isVertical ? {
    x: crossAxis * crossAxisMulti,
    y: mainAxis * mainAxisMulti
  } : {
    x: mainAxis * mainAxisMulti,
    y: crossAxis * crossAxisMulti
  };
}
const offset = function(options) {
  if (options === void 0) {
    options = 0;
  }
  return {
    name: "offset",
    options,
    async fn(state) {
      var _middlewareData$offse, _middlewareData$arrow;
      const {
        x: x2,
        y: y2,
        placement,
        middlewareData
      } = state;
      const diffCoords = await convertValueToCoords(state, options);
      if (placement === ((_middlewareData$offse = middlewareData.offset) == null ? void 0 : _middlewareData$offse.placement) && (_middlewareData$arrow = middlewareData.arrow) != null && _middlewareData$arrow.alignmentOffset) {
        return {};
      }
      return {
        x: x2 + diffCoords.x,
        y: y2 + diffCoords.y,
        data: {
          ...diffCoords,
          placement
        }
      };
    }
  };
};
const shift$1 = function(options) {
  if (options === void 0) {
    options = {};
  }
  return {
    name: "shift",
    options,
    async fn(state) {
      const {
        x: x2,
        y: y2,
        placement
      } = state;
      const {
        mainAxis: checkMainAxis = true,
        crossAxis: checkCrossAxis = false,
        limiter = {
          fn: (_ref) => {
            let {
              x: x3,
              y: y3
            } = _ref;
            return {
              x: x3,
              y: y3
            };
          }
        },
        ...detectOverflowOptions
      } = evaluate(options, state);
      const coords = {
        x: x2,
        y: y2
      };
      const overflow = await detectOverflow(state, detectOverflowOptions);
      const crossAxis = getSideAxis(getSide(placement));
      const mainAxis = getOppositeAxis(crossAxis);
      let mainAxisCoord = coords[mainAxis];
      let crossAxisCoord = coords[crossAxis];
      if (checkMainAxis) {
        const minSide = mainAxis === "y" ? "top" : "left";
        const maxSide = mainAxis === "y" ? "bottom" : "right";
        const min2 = mainAxisCoord + overflow[minSide];
        const max2 = mainAxisCoord - overflow[maxSide];
        mainAxisCoord = clamp(min2, mainAxisCoord, max2);
      }
      if (checkCrossAxis) {
        const minSide = crossAxis === "y" ? "top" : "left";
        const maxSide = crossAxis === "y" ? "bottom" : "right";
        const min2 = crossAxisCoord + overflow[minSide];
        const max2 = crossAxisCoord - overflow[maxSide];
        crossAxisCoord = clamp(min2, crossAxisCoord, max2);
      }
      const limitedCoords = limiter.fn({
        ...state,
        [mainAxis]: mainAxisCoord,
        [crossAxis]: crossAxisCoord
      });
      return {
        ...limitedCoords,
        data: {
          x: limitedCoords.x - x2,
          y: limitedCoords.y - y2
        }
      };
    }
  };
};
const size$1 = function(options) {
  if (options === void 0) {
    options = {};
  }
  return {
    name: "size",
    options,
    async fn(state) {
      const {
        placement,
        rects,
        platform: platform2,
        elements
      } = state;
      const {
        apply = () => {
        },
        ...detectOverflowOptions
      } = evaluate(options, state);
      const overflow = await detectOverflow(state, detectOverflowOptions);
      const side = getSide(placement);
      const alignment = getAlignment(placement);
      const isYAxis = getSideAxis(placement) === "y";
      const {
        width,
        height
      } = rects.floating;
      let heightSide;
      let widthSide;
      if (side === "top" || side === "bottom") {
        heightSide = side;
        widthSide = alignment === (await (platform2.isRTL == null ? void 0 : platform2.isRTL(elements.floating)) ? "start" : "end") ? "left" : "right";
      } else {
        widthSide = side;
        heightSide = alignment === "end" ? "top" : "bottom";
      }
      const overflowAvailableHeight = height - overflow[heightSide];
      const overflowAvailableWidth = width - overflow[widthSide];
      const noShift = !state.middlewareData.shift;
      let availableHeight = overflowAvailableHeight;
      let availableWidth = overflowAvailableWidth;
      if (isYAxis) {
        const maximumClippingWidth = width - overflow.left - overflow.right;
        availableWidth = alignment || noShift ? min(overflowAvailableWidth, maximumClippingWidth) : maximumClippingWidth;
      } else {
        const maximumClippingHeight = height - overflow.top - overflow.bottom;
        availableHeight = alignment || noShift ? min(overflowAvailableHeight, maximumClippingHeight) : maximumClippingHeight;
      }
      if (noShift && !alignment) {
        const xMin = max(overflow.left, 0);
        const xMax = max(overflow.right, 0);
        const yMin = max(overflow.top, 0);
        const yMax = max(overflow.bottom, 0);
        if (isYAxis) {
          availableWidth = width - 2 * (xMin !== 0 || xMax !== 0 ? xMin + xMax : max(overflow.left, overflow.right));
        } else {
          availableHeight = height - 2 * (yMin !== 0 || yMax !== 0 ? yMin + yMax : max(overflow.top, overflow.bottom));
        }
      }
      await apply({
        ...state,
        availableWidth,
        availableHeight
      });
      const nextDimensions = await platform2.getDimensions(elements.floating);
      if (width !== nextDimensions.width || height !== nextDimensions.height) {
        return {
          reset: {
            rects: true
          }
        };
      }
      return {};
    }
  };
};
function getNodeName(node) {
  if (isNode(node)) {
    return (node.nodeName || "").toLowerCase();
  }
  return "#document";
}
function getWindow(node) {
  var _node$ownerDocument;
  return (node == null || (_node$ownerDocument = node.ownerDocument) == null ? void 0 : _node$ownerDocument.defaultView) || window;
}
function getDocumentElement(node) {
  var _ref;
  return (_ref = (isNode(node) ? node.ownerDocument : node.document) || window.document) == null ? void 0 : _ref.documentElement;
}
function isNode(value) {
  return value instanceof Node || value instanceof getWindow(value).Node;
}
function isElement(value) {
  return value instanceof Element || value instanceof getWindow(value).Element;
}
function isHTMLElement(value) {
  return value instanceof HTMLElement || value instanceof getWindow(value).HTMLElement;
}
function isShadowRoot(value) {
  if (typeof ShadowRoot === "undefined") {
    return false;
  }
  return value instanceof ShadowRoot || value instanceof getWindow(value).ShadowRoot;
}
function isOverflowElement(element2) {
  const {
    overflow,
    overflowX,
    overflowY,
    display
  } = getComputedStyle$1(element2);
  return /auto|scroll|overlay|hidden|clip/.test(overflow + overflowY + overflowX) && !["inline", "contents"].includes(display);
}
function isTableElement(element2) {
  return ["table", "td", "th"].includes(getNodeName(element2));
}
function isContainingBlock(element2) {
  const webkit = isWebKit();
  const css = getComputedStyle$1(element2);
  return css.transform !== "none" || css.perspective !== "none" || (css.containerType ? css.containerType !== "normal" : false) || !webkit && (css.backdropFilter ? css.backdropFilter !== "none" : false) || !webkit && (css.filter ? css.filter !== "none" : false) || ["transform", "perspective", "filter"].some((value) => (css.willChange || "").includes(value)) || ["paint", "layout", "strict", "content"].some((value) => (css.contain || "").includes(value));
}
function getContainingBlock(element2) {
  let currentNode = getParentNode(element2);
  while (isHTMLElement(currentNode) && !isLastTraversableNode(currentNode)) {
    if (isContainingBlock(currentNode)) {
      return currentNode;
    } else {
      currentNode = getParentNode(currentNode);
    }
  }
  return null;
}
function isWebKit() {
  if (typeof CSS === "undefined" || !CSS.supports)
    return false;
  return CSS.supports("-webkit-backdrop-filter", "none");
}
function isLastTraversableNode(node) {
  return ["html", "body", "#document"].includes(getNodeName(node));
}
function getComputedStyle$1(element2) {
  return getWindow(element2).getComputedStyle(element2);
}
function getNodeScroll(element2) {
  if (isElement(element2)) {
    return {
      scrollLeft: element2.scrollLeft,
      scrollTop: element2.scrollTop
    };
  }
  return {
    scrollLeft: element2.pageXOffset,
    scrollTop: element2.pageYOffset
  };
}
function getParentNode(node) {
  if (getNodeName(node) === "html") {
    return node;
  }
  const result = (
    // Step into the shadow DOM of the parent of a slotted node.
    node.assignedSlot || // DOM Element detected.
    node.parentNode || // ShadowRoot detected.
    isShadowRoot(node) && node.host || // Fallback.
    getDocumentElement(node)
  );
  return isShadowRoot(result) ? result.host : result;
}
function getNearestOverflowAncestor(node) {
  const parentNode = getParentNode(node);
  if (isLastTraversableNode(parentNode)) {
    return node.ownerDocument ? node.ownerDocument.body : node.body;
  }
  if (isHTMLElement(parentNode) && isOverflowElement(parentNode)) {
    return parentNode;
  }
  return getNearestOverflowAncestor(parentNode);
}
function getOverflowAncestors(node, list2, traverseIframes) {
  var _node$ownerDocument2;
  if (list2 === void 0) {
    list2 = [];
  }
  if (traverseIframes === void 0) {
    traverseIframes = true;
  }
  const scrollableAncestor = getNearestOverflowAncestor(node);
  const isBody = scrollableAncestor === ((_node$ownerDocument2 = node.ownerDocument) == null ? void 0 : _node$ownerDocument2.body);
  const win = getWindow(scrollableAncestor);
  if (isBody) {
    return list2.concat(win, win.visualViewport || [], isOverflowElement(scrollableAncestor) ? scrollableAncestor : [], win.frameElement && traverseIframes ? getOverflowAncestors(win.frameElement) : []);
  }
  return list2.concat(scrollableAncestor, getOverflowAncestors(scrollableAncestor, [], traverseIframes));
}
function getCssDimensions(element2) {
  const css = getComputedStyle$1(element2);
  let width = parseFloat(css.width) || 0;
  let height = parseFloat(css.height) || 0;
  const hasOffset = isHTMLElement(element2);
  const offsetWidth = hasOffset ? element2.offsetWidth : width;
  const offsetHeight = hasOffset ? element2.offsetHeight : height;
  const shouldFallback = round(width) !== offsetWidth || round(height) !== offsetHeight;
  if (shouldFallback) {
    width = offsetWidth;
    height = offsetHeight;
  }
  return {
    width,
    height,
    $: shouldFallback
  };
}
function unwrapElement(element2) {
  return !isElement(element2) ? element2.contextElement : element2;
}
function getScale(element2) {
  const domElement = unwrapElement(element2);
  if (!isHTMLElement(domElement)) {
    return createCoords(1);
  }
  const rect = domElement.getBoundingClientRect();
  const {
    width,
    height,
    $
  } = getCssDimensions(domElement);
  let x2 = ($ ? round(rect.width) : rect.width) / width;
  let y2 = ($ ? round(rect.height) : rect.height) / height;
  if (!x2 || !Number.isFinite(x2)) {
    x2 = 1;
  }
  if (!y2 || !Number.isFinite(y2)) {
    y2 = 1;
  }
  return {
    x: x2,
    y: y2
  };
}
const noOffsets = /* @__PURE__ */ createCoords(0);
function getVisualOffsets(element2) {
  const win = getWindow(element2);
  if (!isWebKit() || !win.visualViewport) {
    return noOffsets;
  }
  return {
    x: win.visualViewport.offsetLeft,
    y: win.visualViewport.offsetTop
  };
}
function shouldAddVisualOffsets(element2, isFixed, floatingOffsetParent) {
  if (isFixed === void 0) {
    isFixed = false;
  }
  if (!floatingOffsetParent || isFixed && floatingOffsetParent !== getWindow(element2)) {
    return false;
  }
  return isFixed;
}
function getBoundingClientRect(element2, includeScale, isFixedStrategy, offsetParent) {
  if (includeScale === void 0) {
    includeScale = false;
  }
  if (isFixedStrategy === void 0) {
    isFixedStrategy = false;
  }
  const clientRect = element2.getBoundingClientRect();
  const domElement = unwrapElement(element2);
  let scale = createCoords(1);
  if (includeScale) {
    if (offsetParent) {
      if (isElement(offsetParent)) {
        scale = getScale(offsetParent);
      }
    } else {
      scale = getScale(element2);
    }
  }
  const visualOffsets = shouldAddVisualOffsets(domElement, isFixedStrategy, offsetParent) ? getVisualOffsets(domElement) : createCoords(0);
  let x2 = (clientRect.left + visualOffsets.x) / scale.x;
  let y2 = (clientRect.top + visualOffsets.y) / scale.y;
  let width = clientRect.width / scale.x;
  let height = clientRect.height / scale.y;
  if (domElement) {
    const win = getWindow(domElement);
    const offsetWin = offsetParent && isElement(offsetParent) ? getWindow(offsetParent) : offsetParent;
    let currentIFrame = win.frameElement;
    while (currentIFrame && offsetParent && offsetWin !== win) {
      const iframeScale = getScale(currentIFrame);
      const iframeRect = currentIFrame.getBoundingClientRect();
      const css = getComputedStyle$1(currentIFrame);
      const left2 = iframeRect.left + (currentIFrame.clientLeft + parseFloat(css.paddingLeft)) * iframeScale.x;
      const top2 = iframeRect.top + (currentIFrame.clientTop + parseFloat(css.paddingTop)) * iframeScale.y;
      x2 *= iframeScale.x;
      y2 *= iframeScale.y;
      width *= iframeScale.x;
      height *= iframeScale.y;
      x2 += left2;
      y2 += top2;
      currentIFrame = getWindow(currentIFrame).frameElement;
    }
  }
  return rectToClientRect({
    width,
    height,
    x: x2,
    y: y2
  });
}
function convertOffsetParentRelativeRectToViewportRelativeRect(_ref) {
  let {
    rect,
    offsetParent,
    strategy
  } = _ref;
  const isOffsetParentAnElement = isHTMLElement(offsetParent);
  const documentElement = getDocumentElement(offsetParent);
  if (offsetParent === documentElement) {
    return rect;
  }
  let scroll = {
    scrollLeft: 0,
    scrollTop: 0
  };
  let scale = createCoords(1);
  const offsets = createCoords(0);
  if (isOffsetParentAnElement || !isOffsetParentAnElement && strategy !== "fixed") {
    if (getNodeName(offsetParent) !== "body" || isOverflowElement(documentElement)) {
      scroll = getNodeScroll(offsetParent);
    }
    if (isHTMLElement(offsetParent)) {
      const offsetRect = getBoundingClientRect(offsetParent);
      scale = getScale(offsetParent);
      offsets.x = offsetRect.x + offsetParent.clientLeft;
      offsets.y = offsetRect.y + offsetParent.clientTop;
    }
  }
  return {
    width: rect.width * scale.x,
    height: rect.height * scale.y,
    x: rect.x * scale.x - scroll.scrollLeft * scale.x + offsets.x,
    y: rect.y * scale.y - scroll.scrollTop * scale.y + offsets.y
  };
}
function getClientRects(element2) {
  return Array.from(element2.getClientRects());
}
function getWindowScrollBarX(element2) {
  return getBoundingClientRect(getDocumentElement(element2)).left + getNodeScroll(element2).scrollLeft;
}
function getDocumentRect(element2) {
  const html = getDocumentElement(element2);
  const scroll = getNodeScroll(element2);
  const body = element2.ownerDocument.body;
  const width = max(html.scrollWidth, html.clientWidth, body.scrollWidth, body.clientWidth);
  const height = max(html.scrollHeight, html.clientHeight, body.scrollHeight, body.clientHeight);
  let x2 = -scroll.scrollLeft + getWindowScrollBarX(element2);
  const y2 = -scroll.scrollTop;
  if (getComputedStyle$1(body).direction === "rtl") {
    x2 += max(html.clientWidth, body.clientWidth) - width;
  }
  return {
    width,
    height,
    x: x2,
    y: y2
  };
}
function getViewportRect(element2, strategy) {
  const win = getWindow(element2);
  const html = getDocumentElement(element2);
  const visualViewport = win.visualViewport;
  let width = html.clientWidth;
  let height = html.clientHeight;
  let x2 = 0;
  let y2 = 0;
  if (visualViewport) {
    width = visualViewport.width;
    height = visualViewport.height;
    const visualViewportBased = isWebKit();
    if (!visualViewportBased || visualViewportBased && strategy === "fixed") {
      x2 = visualViewport.offsetLeft;
      y2 = visualViewport.offsetTop;
    }
  }
  return {
    width,
    height,
    x: x2,
    y: y2
  };
}
function getInnerBoundingClientRect(element2, strategy) {
  const clientRect = getBoundingClientRect(element2, true, strategy === "fixed");
  const top2 = clientRect.top + element2.clientTop;
  const left2 = clientRect.left + element2.clientLeft;
  const scale = isHTMLElement(element2) ? getScale(element2) : createCoords(1);
  const width = element2.clientWidth * scale.x;
  const height = element2.clientHeight * scale.y;
  const x2 = left2 * scale.x;
  const y2 = top2 * scale.y;
  return {
    width,
    height,
    x: x2,
    y: y2
  };
}
function getClientRectFromClippingAncestor(element2, clippingAncestor, strategy) {
  let rect;
  if (clippingAncestor === "viewport") {
    rect = getViewportRect(element2, strategy);
  } else if (clippingAncestor === "document") {
    rect = getDocumentRect(getDocumentElement(element2));
  } else if (isElement(clippingAncestor)) {
    rect = getInnerBoundingClientRect(clippingAncestor, strategy);
  } else {
    const visualOffsets = getVisualOffsets(element2);
    rect = {
      ...clippingAncestor,
      x: clippingAncestor.x - visualOffsets.x,
      y: clippingAncestor.y - visualOffsets.y
    };
  }
  return rectToClientRect(rect);
}
function hasFixedPositionAncestor(element2, stopNode) {
  const parentNode = getParentNode(element2);
  if (parentNode === stopNode || !isElement(parentNode) || isLastTraversableNode(parentNode)) {
    return false;
  }
  return getComputedStyle$1(parentNode).position === "fixed" || hasFixedPositionAncestor(parentNode, stopNode);
}
function getClippingElementAncestors(element2, cache) {
  const cachedResult = cache.get(element2);
  if (cachedResult) {
    return cachedResult;
  }
  let result = getOverflowAncestors(element2, [], false).filter((el) => isElement(el) && getNodeName(el) !== "body");
  let currentContainingBlockComputedStyle = null;
  const elementIsFixed = getComputedStyle$1(element2).position === "fixed";
  let currentNode = elementIsFixed ? getParentNode(element2) : element2;
  while (isElement(currentNode) && !isLastTraversableNode(currentNode)) {
    const computedStyle = getComputedStyle$1(currentNode);
    const currentNodeIsContaining = isContainingBlock(currentNode);
    if (!currentNodeIsContaining && computedStyle.position === "fixed") {
      currentContainingBlockComputedStyle = null;
    }
    const shouldDropCurrentNode = elementIsFixed ? !currentNodeIsContaining && !currentContainingBlockComputedStyle : !currentNodeIsContaining && computedStyle.position === "static" && !!currentContainingBlockComputedStyle && ["absolute", "fixed"].includes(currentContainingBlockComputedStyle.position) || isOverflowElement(currentNode) && !currentNodeIsContaining && hasFixedPositionAncestor(element2, currentNode);
    if (shouldDropCurrentNode) {
      result = result.filter((ancestor) => ancestor !== currentNode);
    } else {
      currentContainingBlockComputedStyle = computedStyle;
    }
    currentNode = getParentNode(currentNode);
  }
  cache.set(element2, result);
  return result;
}
function getClippingRect(_ref) {
  let {
    element: element2,
    boundary,
    rootBoundary,
    strategy
  } = _ref;
  const elementClippingAncestors = boundary === "clippingAncestors" ? getClippingElementAncestors(element2, this._c) : [].concat(boundary);
  const clippingAncestors = [...elementClippingAncestors, rootBoundary];
  const firstClippingAncestor = clippingAncestors[0];
  const clippingRect = clippingAncestors.reduce((accRect, clippingAncestor) => {
    const rect = getClientRectFromClippingAncestor(element2, clippingAncestor, strategy);
    accRect.top = max(rect.top, accRect.top);
    accRect.right = min(rect.right, accRect.right);
    accRect.bottom = min(rect.bottom, accRect.bottom);
    accRect.left = max(rect.left, accRect.left);
    return accRect;
  }, getClientRectFromClippingAncestor(element2, firstClippingAncestor, strategy));
  return {
    width: clippingRect.right - clippingRect.left,
    height: clippingRect.bottom - clippingRect.top,
    x: clippingRect.left,
    y: clippingRect.top
  };
}
function getDimensions(element2) {
  const {
    width,
    height
  } = getCssDimensions(element2);
  return {
    width,
    height
  };
}
function getRectRelativeToOffsetParent(element2, offsetParent, strategy) {
  const isOffsetParentAnElement = isHTMLElement(offsetParent);
  const documentElement = getDocumentElement(offsetParent);
  const isFixed = strategy === "fixed";
  const rect = getBoundingClientRect(element2, true, isFixed, offsetParent);
  let scroll = {
    scrollLeft: 0,
    scrollTop: 0
  };
  const offsets = createCoords(0);
  if (isOffsetParentAnElement || !isOffsetParentAnElement && !isFixed) {
    if (getNodeName(offsetParent) !== "body" || isOverflowElement(documentElement)) {
      scroll = getNodeScroll(offsetParent);
    }
    if (isOffsetParentAnElement) {
      const offsetRect = getBoundingClientRect(offsetParent, true, isFixed, offsetParent);
      offsets.x = offsetRect.x + offsetParent.clientLeft;
      offsets.y = offsetRect.y + offsetParent.clientTop;
    } else if (documentElement) {
      offsets.x = getWindowScrollBarX(documentElement);
    }
  }
  return {
    x: rect.left + scroll.scrollLeft - offsets.x,
    y: rect.top + scroll.scrollTop - offsets.y,
    width: rect.width,
    height: rect.height
  };
}
function getTrueOffsetParent(element2, polyfill2) {
  if (!isHTMLElement(element2) || getComputedStyle$1(element2).position === "fixed") {
    return null;
  }
  if (polyfill2) {
    return polyfill2(element2);
  }
  return element2.offsetParent;
}
function getOffsetParent(element2, polyfill2) {
  const window2 = getWindow(element2);
  if (!isHTMLElement(element2)) {
    return window2;
  }
  let offsetParent = getTrueOffsetParent(element2, polyfill2);
  while (offsetParent && isTableElement(offsetParent) && getComputedStyle$1(offsetParent).position === "static") {
    offsetParent = getTrueOffsetParent(offsetParent, polyfill2);
  }
  if (offsetParent && (getNodeName(offsetParent) === "html" || getNodeName(offsetParent) === "body" && getComputedStyle$1(offsetParent).position === "static" && !isContainingBlock(offsetParent))) {
    return window2;
  }
  return offsetParent || getContainingBlock(element2) || window2;
}
const getElementRects = async function(_ref) {
  let {
    reference,
    floating,
    strategy
  } = _ref;
  const getOffsetParentFn = this.getOffsetParent || getOffsetParent;
  const getDimensionsFn = this.getDimensions;
  return {
    reference: getRectRelativeToOffsetParent(reference, await getOffsetParentFn(floating), strategy),
    floating: {
      x: 0,
      y: 0,
      ...await getDimensionsFn(floating)
    }
  };
};
function isRTL(element2) {
  return getComputedStyle$1(element2).direction === "rtl";
}
const platform$1 = {
  convertOffsetParentRelativeRectToViewportRelativeRect,
  getDocumentElement,
  getClippingRect,
  getOffsetParent,
  getElementRects,
  getClientRects,
  getDimensions,
  getScale,
  isElement,
  isRTL
};
function observeMove(element2, onMove) {
  let io = null;
  let timeoutId;
  const root2 = getDocumentElement(element2);
  function cleanup() {
    clearTimeout(timeoutId);
    io && io.disconnect();
    io = null;
  }
  function refresh2(skip, threshold) {
    if (skip === void 0) {
      skip = false;
    }
    if (threshold === void 0) {
      threshold = 1;
    }
    cleanup();
    const {
      left: left2,
      top: top2,
      width,
      height
    } = element2.getBoundingClientRect();
    if (!skip) {
      onMove();
    }
    if (!width || !height) {
      return;
    }
    const insetTop = floor(top2);
    const insetRight = floor(root2.clientWidth - (left2 + width));
    const insetBottom = floor(root2.clientHeight - (top2 + height));
    const insetLeft = floor(left2);
    const rootMargin = -insetTop + "px " + -insetRight + "px " + -insetBottom + "px " + -insetLeft + "px";
    const options = {
      rootMargin,
      threshold: max(0, min(1, threshold)) || 1
    };
    let isFirstUpdate = true;
    function handleObserve(entries) {
      const ratio = entries[0].intersectionRatio;
      if (ratio !== threshold) {
        if (!isFirstUpdate) {
          return refresh2();
        }
        if (!ratio) {
          timeoutId = setTimeout(() => {
            refresh2(false, 1e-7);
          }, 100);
        } else {
          refresh2(false, ratio);
        }
      }
      isFirstUpdate = false;
    }
    try {
      io = new IntersectionObserver(handleObserve, {
        ...options,
        // Handle <iframe>s
        root: root2.ownerDocument
      });
    } catch (e) {
      io = new IntersectionObserver(handleObserve, options);
    }
    io.observe(element2);
  }
  refresh2(true);
  return cleanup;
}
function autoUpdate(reference, floating, update2, options) {
  if (options === void 0) {
    options = {};
  }
  const {
    ancestorScroll = true,
    ancestorResize = true,
    elementResize = typeof ResizeObserver === "function",
    layoutShift = typeof IntersectionObserver === "function",
    animationFrame = false
  } = options;
  const referenceEl = unwrapElement(reference);
  const ancestors = ancestorScroll || ancestorResize ? [...referenceEl ? getOverflowAncestors(referenceEl) : [], ...getOverflowAncestors(floating)] : [];
  ancestors.forEach((ancestor) => {
    ancestorScroll && ancestor.addEventListener("scroll", update2, {
      passive: true
    });
    ancestorResize && ancestor.addEventListener("resize", update2);
  });
  const cleanupIo = referenceEl && layoutShift ? observeMove(referenceEl, update2) : null;
  let reobserveFrame = -1;
  let resizeObserver = null;
  if (elementResize) {
    resizeObserver = new ResizeObserver((_ref) => {
      let [firstEntry] = _ref;
      if (firstEntry && firstEntry.target === referenceEl && resizeObserver) {
        resizeObserver.unobserve(floating);
        cancelAnimationFrame(reobserveFrame);
        reobserveFrame = requestAnimationFrame(() => {
          resizeObserver && resizeObserver.observe(floating);
        });
      }
      update2();
    });
    if (referenceEl && !animationFrame) {
      resizeObserver.observe(referenceEl);
    }
    resizeObserver.observe(floating);
  }
  let frameId;
  let prevRefRect = animationFrame ? getBoundingClientRect(reference) : null;
  if (animationFrame) {
    frameLoop();
  }
  function frameLoop() {
    const nextRefRect = getBoundingClientRect(reference);
    if (prevRefRect && (nextRefRect.x !== prevRefRect.x || nextRefRect.y !== prevRefRect.y || nextRefRect.width !== prevRefRect.width || nextRefRect.height !== prevRefRect.height)) {
      update2();
    }
    prevRefRect = nextRefRect;
    frameId = requestAnimationFrame(frameLoop);
  }
  update2();
  return () => {
    ancestors.forEach((ancestor) => {
      ancestorScroll && ancestor.removeEventListener("scroll", update2);
      ancestorResize && ancestor.removeEventListener("resize", update2);
    });
    cleanupIo && cleanupIo();
    resizeObserver && resizeObserver.disconnect();
    resizeObserver = null;
    if (animationFrame) {
      cancelAnimationFrame(frameId);
    }
  };
}
const shift = shift$1;
const flip$1 = flip$2;
const size = size$1;
const arrow = arrow$1;
const computePosition = (reference, floating, options) => {
  const cache = /* @__PURE__ */ new Map();
  const mergedOptions = {
    platform: platform$1,
    ...options
  };
  const platformWithCache = {
    ...mergedOptions.platform,
    _c: cache
  };
  return computePosition$1(reference, floating, {
    ...mergedOptions,
    platform: platformWithCache
  });
};
const defaultConfig$1 = {
  strategy: "absolute",
  placement: "top",
  gutter: 5,
  flip: true,
  sameWidth: false,
  overflowPadding: 8
};
const ARROW_TRANSFORM = {
  bottom: "rotate(45deg)",
  left: "rotate(135deg)",
  top: "rotate(225deg)",
  right: "rotate(315deg)"
};
function useFloating(reference, floating, opts = {}) {
  if (!floating || !reference || opts === null)
    return {
      destroy: noop
    };
  const options = { ...defaultConfig$1, ...opts };
  const arrowEl = floating.querySelector("[data-arrow=true]");
  const middleware = [];
  if (options.flip) {
    middleware.push(flip$1({
      boundary: options.boundary,
      padding: options.overflowPadding
    }));
  }
  const arrowOffset = isHTMLElement$1(arrowEl) ? arrowEl.offsetHeight / 2 : 0;
  if (options.gutter || options.offset) {
    const data = options.gutter ? { mainAxis: options.gutter } : options.offset;
    if ((data == null ? void 0 : data.mainAxis) != null) {
      data.mainAxis += arrowOffset;
    }
    middleware.push(offset(data));
  }
  middleware.push(shift({
    boundary: options.boundary,
    crossAxis: options.overlap,
    padding: options.overflowPadding
  }));
  if (arrowEl) {
    middleware.push(arrow({ element: arrowEl, padding: 8 }));
  }
  middleware.push(size({
    padding: options.overflowPadding,
    apply({ rects, availableHeight, availableWidth }) {
      if (options.sameWidth) {
        Object.assign(floating.style, {
          width: `${Math.round(rects.reference.width)}px`,
          minWidth: "unset"
        });
      }
      if (options.fitViewport) {
        Object.assign(floating.style, {
          maxWidth: `${availableWidth}px`,
          maxHeight: `${availableHeight}px`
        });
      }
    }
  }));
  function compute() {
    if (!reference || !floating)
      return;
    const { placement, strategy } = options;
    computePosition(reference, floating, {
      placement,
      middleware,
      strategy
    }).then((data) => {
      const x2 = Math.round(data.x);
      const y2 = Math.round(data.y);
      Object.assign(floating.style, {
        position: options.strategy,
        top: `${y2}px`,
        left: `${x2}px`
      });
      if (isHTMLElement$1(arrowEl) && data.middlewareData.arrow) {
        const { x: x3, y: y3 } = data.middlewareData.arrow;
        const dir = data.placement.split("-")[0];
        Object.assign(arrowEl.style, {
          position: "absolute",
          left: x3 != null ? `${x3}px` : "",
          top: y3 != null ? `${y3}px` : "",
          [dir]: `calc(100% - ${arrowOffset}px)`,
          transform: ARROW_TRANSFORM[dir],
          backgroundColor: "inherit",
          zIndex: "inherit"
        });
      }
      return data;
    });
  }
  Object.assign(floating.style, {
    position: options.strategy
  });
  return {
    destroy: autoUpdate(reference, floating, compute)
  };
}
/*!
* tabbable 6.2.0
* @license MIT, https://github.com/focus-trap/tabbable/blob/master/LICENSE
*/
var candidateSelectors = ["input:not([inert])", "select:not([inert])", "textarea:not([inert])", "a[href]:not([inert])", "button:not([inert])", "[tabindex]:not(slot):not([inert])", "audio[controls]:not([inert])", "video[controls]:not([inert])", '[contenteditable]:not([contenteditable="false"]):not([inert])', "details>summary:first-of-type:not([inert])", "details:not([inert])"];
var candidateSelector = /* @__PURE__ */ candidateSelectors.join(",");
var NoElement = typeof Element === "undefined";
var matches = NoElement ? function() {
} : Element.prototype.matches || Element.prototype.msMatchesSelector || Element.prototype.webkitMatchesSelector;
var getRootNode = !NoElement && Element.prototype.getRootNode ? function(element2) {
  var _element$getRootNode;
  return element2 === null || element2 === void 0 ? void 0 : (_element$getRootNode = element2.getRootNode) === null || _element$getRootNode === void 0 ? void 0 : _element$getRootNode.call(element2);
} : function(element2) {
  return element2 === null || element2 === void 0 ? void 0 : element2.ownerDocument;
};
var isInert = function isInert2(node, lookUp) {
  var _node$getAttribute;
  if (lookUp === void 0) {
    lookUp = true;
  }
  var inertAtt = node === null || node === void 0 ? void 0 : (_node$getAttribute = node.getAttribute) === null || _node$getAttribute === void 0 ? void 0 : _node$getAttribute.call(node, "inert");
  var inert = inertAtt === "" || inertAtt === "true";
  var result = inert || lookUp && node && isInert2(node.parentNode);
  return result;
};
var isContentEditable = function isContentEditable2(node) {
  var _node$getAttribute2;
  var attValue = node === null || node === void 0 ? void 0 : (_node$getAttribute2 = node.getAttribute) === null || _node$getAttribute2 === void 0 ? void 0 : _node$getAttribute2.call(node, "contenteditable");
  return attValue === "" || attValue === "true";
};
var getCandidates = function getCandidates2(el, includeContainer, filter2) {
  if (isInert(el)) {
    return [];
  }
  var candidates = Array.prototype.slice.apply(el.querySelectorAll(candidateSelector));
  if (includeContainer && matches.call(el, candidateSelector)) {
    candidates.unshift(el);
  }
  candidates = candidates.filter(filter2);
  return candidates;
};
var getCandidatesIteratively = function getCandidatesIteratively2(elements, includeContainer, options) {
  var candidates = [];
  var elementsToCheck = Array.from(elements);
  while (elementsToCheck.length) {
    var element2 = elementsToCheck.shift();
    if (isInert(element2, false)) {
      continue;
    }
    if (element2.tagName === "SLOT") {
      var assigned = element2.assignedElements();
      var content = assigned.length ? assigned : element2.children;
      var nestedCandidates = getCandidatesIteratively2(content, true, options);
      if (options.flatten) {
        candidates.push.apply(candidates, nestedCandidates);
      } else {
        candidates.push({
          scopeParent: element2,
          candidates: nestedCandidates
        });
      }
    } else {
      var validCandidate = matches.call(element2, candidateSelector);
      if (validCandidate && options.filter(element2) && (includeContainer || !elements.includes(element2))) {
        candidates.push(element2);
      }
      var shadowRoot = element2.shadowRoot || // check for an undisclosed shadow
      typeof options.getShadowRoot === "function" && options.getShadowRoot(element2);
      var validShadowRoot = !isInert(shadowRoot, false) && (!options.shadowRootFilter || options.shadowRootFilter(element2));
      if (shadowRoot && validShadowRoot) {
        var _nestedCandidates = getCandidatesIteratively2(shadowRoot === true ? element2.children : shadowRoot.children, true, options);
        if (options.flatten) {
          candidates.push.apply(candidates, _nestedCandidates);
        } else {
          candidates.push({
            scopeParent: element2,
            candidates: _nestedCandidates
          });
        }
      } else {
        elementsToCheck.unshift.apply(elementsToCheck, element2.children);
      }
    }
  }
  return candidates;
};
var hasTabIndex = function hasTabIndex2(node) {
  return !isNaN(parseInt(node.getAttribute("tabindex"), 10));
};
var getTabIndex = function getTabIndex2(node) {
  if (!node) {
    throw new Error("No node provided");
  }
  if (node.tabIndex < 0) {
    if ((/^(AUDIO|VIDEO|DETAILS)$/.test(node.tagName) || isContentEditable(node)) && !hasTabIndex(node)) {
      return 0;
    }
  }
  return node.tabIndex;
};
var getSortOrderTabIndex = function getSortOrderTabIndex2(node, isScope) {
  var tabIndex = getTabIndex(node);
  if (tabIndex < 0 && isScope && !hasTabIndex(node)) {
    return 0;
  }
  return tabIndex;
};
var sortOrderedTabbables = function sortOrderedTabbables2(a, b) {
  return a.tabIndex === b.tabIndex ? a.documentOrder - b.documentOrder : a.tabIndex - b.tabIndex;
};
var isInput = function isInput2(node) {
  return node.tagName === "INPUT";
};
var isHiddenInput = function isHiddenInput2(node) {
  return isInput(node) && node.type === "hidden";
};
var isDetailsWithSummary = function isDetailsWithSummary2(node) {
  var r = node.tagName === "DETAILS" && Array.prototype.slice.apply(node.children).some(function(child) {
    return child.tagName === "SUMMARY";
  });
  return r;
};
var getCheckedRadio = function getCheckedRadio2(nodes, form) {
  for (var i2 = 0; i2 < nodes.length; i2++) {
    if (nodes[i2].checked && nodes[i2].form === form) {
      return nodes[i2];
    }
  }
};
var isTabbableRadio = function isTabbableRadio2(node) {
  if (!node.name) {
    return true;
  }
  var radioScope = node.form || getRootNode(node);
  var queryRadios = function queryRadios2(name2) {
    return radioScope.querySelectorAll('input[type="radio"][name="' + name2 + '"]');
  };
  var radioSet;
  if (typeof window !== "undefined" && typeof window.CSS !== "undefined" && typeof window.CSS.escape === "function") {
    radioSet = queryRadios(window.CSS.escape(node.name));
  } else {
    try {
      radioSet = queryRadios(node.name);
    } catch (err) {
      console.error("Looks like you have a radio button with a name attribute containing invalid CSS selector characters and need the CSS.escape polyfill: %s", err.message);
      return false;
    }
  }
  var checked = getCheckedRadio(radioSet, node.form);
  return !checked || checked === node;
};
var isRadio = function isRadio2(node) {
  return isInput(node) && node.type === "radio";
};
var isNonTabbableRadio = function isNonTabbableRadio2(node) {
  return isRadio(node) && !isTabbableRadio(node);
};
var isNodeAttached = function isNodeAttached2(node) {
  var _nodeRoot;
  var nodeRoot = node && getRootNode(node);
  var nodeRootHost = (_nodeRoot = nodeRoot) === null || _nodeRoot === void 0 ? void 0 : _nodeRoot.host;
  var attached = false;
  if (nodeRoot && nodeRoot !== node) {
    var _nodeRootHost, _nodeRootHost$ownerDo, _node$ownerDocument;
    attached = !!((_nodeRootHost = nodeRootHost) !== null && _nodeRootHost !== void 0 && (_nodeRootHost$ownerDo = _nodeRootHost.ownerDocument) !== null && _nodeRootHost$ownerDo !== void 0 && _nodeRootHost$ownerDo.contains(nodeRootHost) || node !== null && node !== void 0 && (_node$ownerDocument = node.ownerDocument) !== null && _node$ownerDocument !== void 0 && _node$ownerDocument.contains(node));
    while (!attached && nodeRootHost) {
      var _nodeRoot2, _nodeRootHost2, _nodeRootHost2$ownerD;
      nodeRoot = getRootNode(nodeRootHost);
      nodeRootHost = (_nodeRoot2 = nodeRoot) === null || _nodeRoot2 === void 0 ? void 0 : _nodeRoot2.host;
      attached = !!((_nodeRootHost2 = nodeRootHost) !== null && _nodeRootHost2 !== void 0 && (_nodeRootHost2$ownerD = _nodeRootHost2.ownerDocument) !== null && _nodeRootHost2$ownerD !== void 0 && _nodeRootHost2$ownerD.contains(nodeRootHost));
    }
  }
  return attached;
};
var isZeroArea = function isZeroArea2(node) {
  var _node$getBoundingClie = node.getBoundingClientRect(), width = _node$getBoundingClie.width, height = _node$getBoundingClie.height;
  return width === 0 && height === 0;
};
var isHidden = function isHidden2(node, _ref) {
  var displayCheck = _ref.displayCheck, getShadowRoot = _ref.getShadowRoot;
  if (getComputedStyle(node).visibility === "hidden") {
    return true;
  }
  var isDirectSummary = matches.call(node, "details>summary:first-of-type");
  var nodeUnderDetails = isDirectSummary ? node.parentElement : node;
  if (matches.call(nodeUnderDetails, "details:not([open]) *")) {
    return true;
  }
  if (!displayCheck || displayCheck === "full" || displayCheck === "legacy-full") {
    if (typeof getShadowRoot === "function") {
      var originalNode = node;
      while (node) {
        var parentElement = node.parentElement;
        var rootNode = getRootNode(node);
        if (parentElement && !parentElement.shadowRoot && getShadowRoot(parentElement) === true) {
          return isZeroArea(node);
        } else if (node.assignedSlot) {
          node = node.assignedSlot;
        } else if (!parentElement && rootNode !== node.ownerDocument) {
          node = rootNode.host;
        } else {
          node = parentElement;
        }
      }
      node = originalNode;
    }
    if (isNodeAttached(node)) {
      return !node.getClientRects().length;
    }
    if (displayCheck !== "legacy-full") {
      return true;
    }
  } else if (displayCheck === "non-zero-area") {
    return isZeroArea(node);
  }
  return false;
};
var isDisabledFromFieldset = function isDisabledFromFieldset2(node) {
  if (/^(INPUT|BUTTON|SELECT|TEXTAREA)$/.test(node.tagName)) {
    var parentNode = node.parentElement;
    while (parentNode) {
      if (parentNode.tagName === "FIELDSET" && parentNode.disabled) {
        for (var i2 = 0; i2 < parentNode.children.length; i2++) {
          var child = parentNode.children.item(i2);
          if (child.tagName === "LEGEND") {
            return matches.call(parentNode, "fieldset[disabled] *") ? true : !child.contains(node);
          }
        }
        return true;
      }
      parentNode = parentNode.parentElement;
    }
  }
  return false;
};
var isNodeMatchingSelectorFocusable = function isNodeMatchingSelectorFocusable2(options, node) {
  if (node.disabled || // we must do an inert look up to filter out any elements inside an inert ancestor
  //  because we're limited in the type of selectors we can use in JSDom (see related
  //  note related to `candidateSelectors`)
  isInert(node) || isHiddenInput(node) || isHidden(node, options) || // For a details element with a summary, the summary element gets the focus
  isDetailsWithSummary(node) || isDisabledFromFieldset(node)) {
    return false;
  }
  return true;
};
var isNodeMatchingSelectorTabbable = function isNodeMatchingSelectorTabbable2(options, node) {
  if (isNonTabbableRadio(node) || getTabIndex(node) < 0 || !isNodeMatchingSelectorFocusable(options, node)) {
    return false;
  }
  return true;
};
var isValidShadowRootTabbable = function isValidShadowRootTabbable2(shadowHostNode) {
  var tabIndex = parseInt(shadowHostNode.getAttribute("tabindex"), 10);
  if (isNaN(tabIndex) || tabIndex >= 0) {
    return true;
  }
  return false;
};
var sortByOrder = function sortByOrder2(candidates) {
  var regularTabbables = [];
  var orderedTabbables = [];
  candidates.forEach(function(item, i2) {
    var isScope = !!item.scopeParent;
    var element2 = isScope ? item.scopeParent : item;
    var candidateTabindex = getSortOrderTabIndex(element2, isScope);
    var elements = isScope ? sortByOrder2(item.candidates) : element2;
    if (candidateTabindex === 0) {
      isScope ? regularTabbables.push.apply(regularTabbables, elements) : regularTabbables.push(element2);
    } else {
      orderedTabbables.push({
        documentOrder: i2,
        tabIndex: candidateTabindex,
        item,
        isScope,
        content: elements
      });
    }
  });
  return orderedTabbables.sort(sortOrderedTabbables).reduce(function(acc, sortable) {
    sortable.isScope ? acc.push.apply(acc, sortable.content) : acc.push(sortable.content);
    return acc;
  }, []).concat(regularTabbables);
};
var tabbable = function tabbable2(container, options) {
  options = options || {};
  var candidates;
  if (options.getShadowRoot) {
    candidates = getCandidatesIteratively([container], options.includeContainer, {
      filter: isNodeMatchingSelectorTabbable.bind(null, options),
      flatten: false,
      getShadowRoot: options.getShadowRoot,
      shadowRootFilter: isValidShadowRootTabbable
    });
  } else {
    candidates = getCandidates(container, options.includeContainer, isNodeMatchingSelectorTabbable.bind(null, options));
  }
  return sortByOrder(candidates);
};
var focusable = function focusable2(container, options) {
  options = options || {};
  var candidates;
  if (options.getShadowRoot) {
    candidates = getCandidatesIteratively([container], options.includeContainer, {
      filter: isNodeMatchingSelectorFocusable.bind(null, options),
      flatten: true,
      getShadowRoot: options.getShadowRoot
    });
  } else {
    candidates = getCandidates(container, options.includeContainer, isNodeMatchingSelectorFocusable.bind(null, options));
  }
  return candidates;
};
var isTabbable = function isTabbable2(node, options) {
  options = options || {};
  if (!node) {
    throw new Error("No node provided");
  }
  if (matches.call(node, candidateSelector) === false) {
    return false;
  }
  return isNodeMatchingSelectorTabbable(options, node);
};
var focusableCandidateSelector = /* @__PURE__ */ candidateSelectors.concat("iframe").join(",");
var isFocusable = function isFocusable2(node, options) {
  options = options || {};
  if (!node) {
    throw new Error("No node provided");
  }
  if (matches.call(node, focusableCandidateSelector) === false) {
    return false;
  }
  return isNodeMatchingSelectorFocusable(options, node);
};
/*!
* focus-trap 7.5.4
* @license MIT, https://github.com/focus-trap/focus-trap/blob/master/LICENSE
*/
function ownKeys(e, r) {
  var t = Object.keys(e);
  if (Object.getOwnPropertySymbols) {
    var o = Object.getOwnPropertySymbols(e);
    r && (o = o.filter(function(r2) {
      return Object.getOwnPropertyDescriptor(e, r2).enumerable;
    })), t.push.apply(t, o);
  }
  return t;
}
function _objectSpread2(e) {
  for (var r = 1; r < arguments.length; r++) {
    var t = null != arguments[r] ? arguments[r] : {};
    r % 2 ? ownKeys(Object(t), true).forEach(function(r2) {
      _defineProperty(e, r2, t[r2]);
    }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function(r2) {
      Object.defineProperty(e, r2, Object.getOwnPropertyDescriptor(t, r2));
    });
  }
  return e;
}
function _defineProperty(obj, key2, value) {
  key2 = _toPropertyKey(key2);
  if (key2 in obj) {
    Object.defineProperty(obj, key2, {
      value,
      enumerable: true,
      configurable: true,
      writable: true
    });
  } else {
    obj[key2] = value;
  }
  return obj;
}
function _toPrimitive(input, hint) {
  if (typeof input !== "object" || input === null)
    return input;
  var prim = input[Symbol.toPrimitive];
  if (prim !== void 0) {
    var res = prim.call(input, hint || "default");
    if (typeof res !== "object")
      return res;
    throw new TypeError("@@toPrimitive must return a primitive value.");
  }
  return (hint === "string" ? String : Number)(input);
}
function _toPropertyKey(arg) {
  var key2 = _toPrimitive(arg, "string");
  return typeof key2 === "symbol" ? key2 : String(key2);
}
var activeFocusTraps = {
  activateTrap: function activateTrap(trapStack, trap) {
    if (trapStack.length > 0) {
      var activeTrap = trapStack[trapStack.length - 1];
      if (activeTrap !== trap) {
        activeTrap.pause();
      }
    }
    var trapIndex = trapStack.indexOf(trap);
    if (trapIndex === -1) {
      trapStack.push(trap);
    } else {
      trapStack.splice(trapIndex, 1);
      trapStack.push(trap);
    }
  },
  deactivateTrap: function deactivateTrap(trapStack, trap) {
    var trapIndex = trapStack.indexOf(trap);
    if (trapIndex !== -1) {
      trapStack.splice(trapIndex, 1);
    }
    if (trapStack.length > 0) {
      trapStack[trapStack.length - 1].unpause();
    }
  }
};
var isSelectableInput = function isSelectableInput2(node) {
  return node.tagName && node.tagName.toLowerCase() === "input" && typeof node.select === "function";
};
var isEscapeEvent = function isEscapeEvent2(e) {
  return (e === null || e === void 0 ? void 0 : e.key) === "Escape" || (e === null || e === void 0 ? void 0 : e.key) === "Esc" || (e === null || e === void 0 ? void 0 : e.keyCode) === 27;
};
var isTabEvent = function isTabEvent2(e) {
  return (e === null || e === void 0 ? void 0 : e.key) === "Tab" || (e === null || e === void 0 ? void 0 : e.keyCode) === 9;
};
var isKeyForward = function isKeyForward2(e) {
  return isTabEvent(e) && !e.shiftKey;
};
var isKeyBackward = function isKeyBackward2(e) {
  return isTabEvent(e) && e.shiftKey;
};
var delay = function delay2(fn) {
  return setTimeout(fn, 0);
};
var findIndex = function findIndex2(arr, fn) {
  var idx = -1;
  arr.every(function(value, i2) {
    if (fn(value)) {
      idx = i2;
      return false;
    }
    return true;
  });
  return idx;
};
var valueOrHandler = function valueOrHandler2(value) {
  for (var _len = arguments.length, params = new Array(_len > 1 ? _len - 1 : 0), _key = 1; _key < _len; _key++) {
    params[_key - 1] = arguments[_key];
  }
  return typeof value === "function" ? value.apply(void 0, params) : value;
};
var getActualTarget = function getActualTarget2(event) {
  return event.target.shadowRoot && typeof event.composedPath === "function" ? event.composedPath()[0] : event.target;
};
var internalTrapStack = [];
var createFocusTrap$1 = function createFocusTrap(elements, userOptions) {
  var doc = (userOptions === null || userOptions === void 0 ? void 0 : userOptions.document) || document;
  var trapStack = (userOptions === null || userOptions === void 0 ? void 0 : userOptions.trapStack) || internalTrapStack;
  var config = _objectSpread2({
    returnFocusOnDeactivate: true,
    escapeDeactivates: true,
    delayInitialFocus: true,
    isKeyForward,
    isKeyBackward
  }, userOptions);
  var state = {
    // containers given to createFocusTrap()
    // @type {Array<HTMLElement>}
    containers: [],
    // list of objects identifying tabbable nodes in `containers` in the trap
    // NOTE: it's possible that a group has no tabbable nodes if nodes get removed while the trap
    //  is active, but the trap should never get to a state where there isn't at least one group
    //  with at least one tabbable node in it (that would lead to an error condition that would
    //  result in an error being thrown)
    // @type {Array<{
    //   container: HTMLElement,
    //   tabbableNodes: Array<HTMLElement>, // empty if none
    //   focusableNodes: Array<HTMLElement>, // empty if none
    //   posTabIndexesFound: boolean,
    //   firstTabbableNode: HTMLElement|undefined,
    //   lastTabbableNode: HTMLElement|undefined,
    //   firstDomTabbableNode: HTMLElement|undefined,
    //   lastDomTabbableNode: HTMLElement|undefined,
    //   nextTabbableNode: (node: HTMLElement, forward: boolean) => HTMLElement|undefined
    // }>}
    containerGroups: [],
    // same order/length as `containers` list
    // references to objects in `containerGroups`, but only those that actually have
    //  tabbable nodes in them
    // NOTE: same order as `containers` and `containerGroups`, but __not necessarily__
    //  the same length
    tabbableGroups: [],
    nodeFocusedBeforeActivation: null,
    mostRecentlyFocusedNode: null,
    active: false,
    paused: false,
    // timer ID for when delayInitialFocus is true and initial focus in this trap
    //  has been delayed during activation
    delayInitialFocusTimer: void 0,
    // the most recent KeyboardEvent for the configured nav key (typically [SHIFT+]TAB), if any
    recentNavEvent: void 0
  };
  var trap;
  var getOption = function getOption2(configOverrideOptions, optionName, configOptionName) {
    return configOverrideOptions && configOverrideOptions[optionName] !== void 0 ? configOverrideOptions[optionName] : config[configOptionName || optionName];
  };
  var findContainerIndex = function findContainerIndex2(element2, event) {
    var composedPath = typeof (event === null || event === void 0 ? void 0 : event.composedPath) === "function" ? event.composedPath() : void 0;
    return state.containerGroups.findIndex(function(_ref) {
      var container = _ref.container, tabbableNodes = _ref.tabbableNodes;
      return container.contains(element2) || // fall back to explicit tabbable search which will take into consideration any
      //  web components if the `tabbableOptions.getShadowRoot` option was used for
      //  the trap, enabling shadow DOM support in tabbable (`Node.contains()` doesn't
      //  look inside web components even if open)
      (composedPath === null || composedPath === void 0 ? void 0 : composedPath.includes(container)) || tabbableNodes.find(function(node) {
        return node === element2;
      });
    });
  };
  var getNodeForOption = function getNodeForOption2(optionName) {
    var optionValue = config[optionName];
    if (typeof optionValue === "function") {
      for (var _len2 = arguments.length, params = new Array(_len2 > 1 ? _len2 - 1 : 0), _key2 = 1; _key2 < _len2; _key2++) {
        params[_key2 - 1] = arguments[_key2];
      }
      optionValue = optionValue.apply(void 0, params);
    }
    if (optionValue === true) {
      optionValue = void 0;
    }
    if (!optionValue) {
      if (optionValue === void 0 || optionValue === false) {
        return optionValue;
      }
      throw new Error("`".concat(optionName, "` was specified but was not a node, or did not return a node"));
    }
    var node = optionValue;
    if (typeof optionValue === "string") {
      node = doc.querySelector(optionValue);
      if (!node) {
        throw new Error("`".concat(optionName, "` as selector refers to no known node"));
      }
    }
    return node;
  };
  var getInitialFocusNode = function getInitialFocusNode2() {
    var node = getNodeForOption("initialFocus");
    if (node === false) {
      return false;
    }
    if (node === void 0 || !isFocusable(node, config.tabbableOptions)) {
      if (findContainerIndex(doc.activeElement) >= 0) {
        node = doc.activeElement;
      } else {
        var firstTabbableGroup = state.tabbableGroups[0];
        var firstTabbableNode = firstTabbableGroup && firstTabbableGroup.firstTabbableNode;
        node = firstTabbableNode || getNodeForOption("fallbackFocus");
      }
    }
    if (!node) {
      throw new Error("Your focus-trap needs to have at least one focusable element");
    }
    return node;
  };
  var updateTabbableNodes = function updateTabbableNodes2() {
    state.containerGroups = state.containers.map(function(container) {
      var tabbableNodes = tabbable(container, config.tabbableOptions);
      var focusableNodes = focusable(container, config.tabbableOptions);
      var firstTabbableNode = tabbableNodes.length > 0 ? tabbableNodes[0] : void 0;
      var lastTabbableNode = tabbableNodes.length > 0 ? tabbableNodes[tabbableNodes.length - 1] : void 0;
      var firstDomTabbableNode = focusableNodes.find(function(node) {
        return isTabbable(node);
      });
      var lastDomTabbableNode = focusableNodes.slice().reverse().find(function(node) {
        return isTabbable(node);
      });
      var posTabIndexesFound = !!tabbableNodes.find(function(node) {
        return getTabIndex(node) > 0;
      });
      return {
        container,
        tabbableNodes,
        focusableNodes,
        /** True if at least one node with positive `tabindex` was found in this container. */
        posTabIndexesFound,
        /** First tabbable node in container, __tabindex__ order; `undefined` if none. */
        firstTabbableNode,
        /** Last tabbable node in container, __tabindex__ order; `undefined` if none. */
        lastTabbableNode,
        // NOTE: DOM order is NOT NECESSARILY "document position" order, but figuring that out
        //  would require more than just https://developer.mozilla.org/en-US/docs/Web/API/Node/compareDocumentPosition
        //  because that API doesn't work with Shadow DOM as well as it should (@see
        //  https://github.com/whatwg/dom/issues/320) and since this first/last is only needed, so far,
        //  to address an edge case related to positive tabindex support, this seems like a much easier,
        //  "close enough most of the time" alternative for positive tabindexes which should generally
        //  be avoided anyway...
        /** First tabbable node in container, __DOM__ order; `undefined` if none. */
        firstDomTabbableNode,
        /** Last tabbable node in container, __DOM__ order; `undefined` if none. */
        lastDomTabbableNode,
        /**
         * Finds the __tabbable__ node that follows the given node in the specified direction,
         *  in this container, if any.
         * @param {HTMLElement} node
         * @param {boolean} [forward] True if going in forward tab order; false if going
         *  in reverse.
         * @returns {HTMLElement|undefined} The next tabbable node, if any.
         */
        nextTabbableNode: function nextTabbableNode(node) {
          var forward2 = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : true;
          var nodeIdx = tabbableNodes.indexOf(node);
          if (nodeIdx < 0) {
            if (forward2) {
              return focusableNodes.slice(focusableNodes.indexOf(node) + 1).find(function(el) {
                return isTabbable(el);
              });
            }
            return focusableNodes.slice(0, focusableNodes.indexOf(node)).reverse().find(function(el) {
              return isTabbable(el);
            });
          }
          return tabbableNodes[nodeIdx + (forward2 ? 1 : -1)];
        }
      };
    });
    state.tabbableGroups = state.containerGroups.filter(function(group2) {
      return group2.tabbableNodes.length > 0;
    });
    if (state.tabbableGroups.length <= 0 && !getNodeForOption("fallbackFocus")) {
      throw new Error("Your focus-trap must have at least one container with at least one tabbable node in it at all times");
    }
    if (state.containerGroups.find(function(g) {
      return g.posTabIndexesFound;
    }) && state.containerGroups.length > 1) {
      throw new Error("At least one node with a positive tabindex was found in one of your focus-trap's multiple containers. Positive tabindexes are only supported in single-container focus-traps.");
    }
  };
  var getActiveElement = function getActiveElement2(el) {
    var activeElement = el.activeElement;
    if (!activeElement) {
      return;
    }
    if (activeElement.shadowRoot && activeElement.shadowRoot.activeElement !== null) {
      return getActiveElement2(activeElement.shadowRoot);
    }
    return activeElement;
  };
  var tryFocus = function tryFocus2(node) {
    if (node === false) {
      return;
    }
    if (node === getActiveElement(document)) {
      return;
    }
    if (!node || !node.focus) {
      tryFocus2(getInitialFocusNode());
      return;
    }
    node.focus({
      preventScroll: !!config.preventScroll
    });
    state.mostRecentlyFocusedNode = node;
    if (isSelectableInput(node)) {
      node.select();
    }
  };
  var getReturnFocusNode = function getReturnFocusNode2(previousActiveElement) {
    var node = getNodeForOption("setReturnFocus", previousActiveElement);
    return node ? node : node === false ? false : previousActiveElement;
  };
  var findNextNavNode = function findNextNavNode2(_ref2) {
    var target = _ref2.target, event = _ref2.event, _ref2$isBackward = _ref2.isBackward, isBackward = _ref2$isBackward === void 0 ? false : _ref2$isBackward;
    target = target || getActualTarget(event);
    updateTabbableNodes();
    var destinationNode = null;
    if (state.tabbableGroups.length > 0) {
      var containerIndex = findContainerIndex(target, event);
      var containerGroup = containerIndex >= 0 ? state.containerGroups[containerIndex] : void 0;
      if (containerIndex < 0) {
        if (isBackward) {
          destinationNode = state.tabbableGroups[state.tabbableGroups.length - 1].lastTabbableNode;
        } else {
          destinationNode = state.tabbableGroups[0].firstTabbableNode;
        }
      } else if (isBackward) {
        var startOfGroupIndex = findIndex(state.tabbableGroups, function(_ref3) {
          var firstTabbableNode = _ref3.firstTabbableNode;
          return target === firstTabbableNode;
        });
        if (startOfGroupIndex < 0 && (containerGroup.container === target || isFocusable(target, config.tabbableOptions) && !isTabbable(target, config.tabbableOptions) && !containerGroup.nextTabbableNode(target, false))) {
          startOfGroupIndex = containerIndex;
        }
        if (startOfGroupIndex >= 0) {
          var destinationGroupIndex = startOfGroupIndex === 0 ? state.tabbableGroups.length - 1 : startOfGroupIndex - 1;
          var destinationGroup = state.tabbableGroups[destinationGroupIndex];
          destinationNode = getTabIndex(target) >= 0 ? destinationGroup.lastTabbableNode : destinationGroup.lastDomTabbableNode;
        } else if (!isTabEvent(event)) {
          destinationNode = containerGroup.nextTabbableNode(target, false);
        }
      } else {
        var lastOfGroupIndex = findIndex(state.tabbableGroups, function(_ref4) {
          var lastTabbableNode = _ref4.lastTabbableNode;
          return target === lastTabbableNode;
        });
        if (lastOfGroupIndex < 0 && (containerGroup.container === target || isFocusable(target, config.tabbableOptions) && !isTabbable(target, config.tabbableOptions) && !containerGroup.nextTabbableNode(target))) {
          lastOfGroupIndex = containerIndex;
        }
        if (lastOfGroupIndex >= 0) {
          var _destinationGroupIndex = lastOfGroupIndex === state.tabbableGroups.length - 1 ? 0 : lastOfGroupIndex + 1;
          var _destinationGroup = state.tabbableGroups[_destinationGroupIndex];
          destinationNode = getTabIndex(target) >= 0 ? _destinationGroup.firstTabbableNode : _destinationGroup.firstDomTabbableNode;
        } else if (!isTabEvent(event)) {
          destinationNode = containerGroup.nextTabbableNode(target);
        }
      }
    } else {
      destinationNode = getNodeForOption("fallbackFocus");
    }
    return destinationNode;
  };
  var checkPointerDown = function checkPointerDown2(e) {
    var target = getActualTarget(e);
    if (findContainerIndex(target, e) >= 0) {
      return;
    }
    if (valueOrHandler(config.clickOutsideDeactivates, e)) {
      trap.deactivate({
        // NOTE: by setting `returnFocus: false`, deactivate() will do nothing,
        //  which will result in the outside click setting focus to the node
        //  that was clicked (and if not focusable, to "nothing"); by setting
        //  `returnFocus: true`, we'll attempt to re-focus the node originally-focused
        //  on activation (or the configured `setReturnFocus` node), whether the
        //  outside click was on a focusable node or not
        returnFocus: config.returnFocusOnDeactivate
      });
      return;
    }
    if (valueOrHandler(config.allowOutsideClick, e)) {
      return;
    }
    e.preventDefault();
  };
  var checkFocusIn = function checkFocusIn2(event) {
    var target = getActualTarget(event);
    var targetContained = findContainerIndex(target, event) >= 0;
    if (targetContained || target instanceof Document) {
      if (targetContained) {
        state.mostRecentlyFocusedNode = target;
      }
    } else {
      event.stopImmediatePropagation();
      var nextNode;
      var navAcrossContainers = true;
      if (state.mostRecentlyFocusedNode) {
        if (getTabIndex(state.mostRecentlyFocusedNode) > 0) {
          var mruContainerIdx = findContainerIndex(state.mostRecentlyFocusedNode);
          var tabbableNodes = state.containerGroups[mruContainerIdx].tabbableNodes;
          if (tabbableNodes.length > 0) {
            var mruTabIdx = tabbableNodes.findIndex(function(node) {
              return node === state.mostRecentlyFocusedNode;
            });
            if (mruTabIdx >= 0) {
              if (config.isKeyForward(state.recentNavEvent)) {
                if (mruTabIdx + 1 < tabbableNodes.length) {
                  nextNode = tabbableNodes[mruTabIdx + 1];
                  navAcrossContainers = false;
                }
              } else {
                if (mruTabIdx - 1 >= 0) {
                  nextNode = tabbableNodes[mruTabIdx - 1];
                  navAcrossContainers = false;
                }
              }
            }
          }
        } else {
          if (!state.containerGroups.some(function(g) {
            return g.tabbableNodes.some(function(n) {
              return getTabIndex(n) > 0;
            });
          })) {
            navAcrossContainers = false;
          }
        }
      } else {
        navAcrossContainers = false;
      }
      if (navAcrossContainers) {
        nextNode = findNextNavNode({
          // move FROM the MRU node, not event-related node (which will be the node that is
          //  outside the trap causing the focus escape we're trying to fix)
          target: state.mostRecentlyFocusedNode,
          isBackward: config.isKeyBackward(state.recentNavEvent)
        });
      }
      if (nextNode) {
        tryFocus(nextNode);
      } else {
        tryFocus(state.mostRecentlyFocusedNode || getInitialFocusNode());
      }
    }
    state.recentNavEvent = void 0;
  };
  var checkKeyNav = function checkKeyNav2(event) {
    var isBackward = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : false;
    state.recentNavEvent = event;
    var destinationNode = findNextNavNode({
      event,
      isBackward
    });
    if (destinationNode) {
      if (isTabEvent(event)) {
        event.preventDefault();
      }
      tryFocus(destinationNode);
    }
  };
  var checkKey = function checkKey2(event) {
    if (isEscapeEvent(event) && valueOrHandler(config.escapeDeactivates, event) !== false) {
      event.preventDefault();
      trap.deactivate();
      return;
    }
    if (config.isKeyForward(event) || config.isKeyBackward(event)) {
      checkKeyNav(event, config.isKeyBackward(event));
    }
  };
  var checkClick = function checkClick2(e) {
    var target = getActualTarget(e);
    if (findContainerIndex(target, e) >= 0) {
      return;
    }
    if (valueOrHandler(config.clickOutsideDeactivates, e)) {
      return;
    }
    if (valueOrHandler(config.allowOutsideClick, e)) {
      return;
    }
    e.preventDefault();
    e.stopImmediatePropagation();
  };
  var addListeners = function addListeners2() {
    if (!state.active) {
      return;
    }
    activeFocusTraps.activateTrap(trapStack, trap);
    state.delayInitialFocusTimer = config.delayInitialFocus ? delay(function() {
      tryFocus(getInitialFocusNode());
    }) : tryFocus(getInitialFocusNode());
    doc.addEventListener("focusin", checkFocusIn, true);
    doc.addEventListener("mousedown", checkPointerDown, {
      capture: true,
      passive: false
    });
    doc.addEventListener("touchstart", checkPointerDown, {
      capture: true,
      passive: false
    });
    doc.addEventListener("click", checkClick, {
      capture: true,
      passive: false
    });
    doc.addEventListener("keydown", checkKey, {
      capture: true,
      passive: false
    });
    return trap;
  };
  var removeListeners = function removeListeners2() {
    if (!state.active) {
      return;
    }
    doc.removeEventListener("focusin", checkFocusIn, true);
    doc.removeEventListener("mousedown", checkPointerDown, true);
    doc.removeEventListener("touchstart", checkPointerDown, true);
    doc.removeEventListener("click", checkClick, true);
    doc.removeEventListener("keydown", checkKey, true);
    return trap;
  };
  var checkDomRemoval = function checkDomRemoval2(mutations) {
    var isFocusedNodeRemoved = mutations.some(function(mutation) {
      var removedNodes = Array.from(mutation.removedNodes);
      return removedNodes.some(function(node) {
        return node === state.mostRecentlyFocusedNode;
      });
    });
    if (isFocusedNodeRemoved) {
      tryFocus(getInitialFocusNode());
    }
  };
  var mutationObserver = typeof window !== "undefined" && "MutationObserver" in window ? new MutationObserver(checkDomRemoval) : void 0;
  var updateObservedNodes = function updateObservedNodes2() {
    if (!mutationObserver) {
      return;
    }
    mutationObserver.disconnect();
    if (state.active && !state.paused) {
      state.containers.map(function(container) {
        mutationObserver.observe(container, {
          subtree: true,
          childList: true
        });
      });
    }
  };
  trap = {
    get active() {
      return state.active;
    },
    get paused() {
      return state.paused;
    },
    activate: function activate(activateOptions) {
      if (state.active) {
        return this;
      }
      var onActivate = getOption(activateOptions, "onActivate");
      var onPostActivate = getOption(activateOptions, "onPostActivate");
      var checkCanFocusTrap = getOption(activateOptions, "checkCanFocusTrap");
      if (!checkCanFocusTrap) {
        updateTabbableNodes();
      }
      state.active = true;
      state.paused = false;
      state.nodeFocusedBeforeActivation = doc.activeElement;
      onActivate === null || onActivate === void 0 || onActivate();
      var finishActivation = function finishActivation2() {
        if (checkCanFocusTrap) {
          updateTabbableNodes();
        }
        addListeners();
        updateObservedNodes();
        onPostActivate === null || onPostActivate === void 0 || onPostActivate();
      };
      if (checkCanFocusTrap) {
        checkCanFocusTrap(state.containers.concat()).then(finishActivation, finishActivation);
        return this;
      }
      finishActivation();
      return this;
    },
    deactivate: function deactivate(deactivateOptions) {
      if (!state.active) {
        return this;
      }
      var options = _objectSpread2({
        onDeactivate: config.onDeactivate,
        onPostDeactivate: config.onPostDeactivate,
        checkCanReturnFocus: config.checkCanReturnFocus
      }, deactivateOptions);
      clearTimeout(state.delayInitialFocusTimer);
      state.delayInitialFocusTimer = void 0;
      removeListeners();
      state.active = false;
      state.paused = false;
      updateObservedNodes();
      activeFocusTraps.deactivateTrap(trapStack, trap);
      var onDeactivate = getOption(options, "onDeactivate");
      var onPostDeactivate = getOption(options, "onPostDeactivate");
      var checkCanReturnFocus = getOption(options, "checkCanReturnFocus");
      var returnFocus = getOption(options, "returnFocus", "returnFocusOnDeactivate");
      onDeactivate === null || onDeactivate === void 0 || onDeactivate();
      var finishDeactivation = function finishDeactivation2() {
        delay(function() {
          if (returnFocus) {
            tryFocus(getReturnFocusNode(state.nodeFocusedBeforeActivation));
          }
          onPostDeactivate === null || onPostDeactivate === void 0 || onPostDeactivate();
        });
      };
      if (returnFocus && checkCanReturnFocus) {
        checkCanReturnFocus(getReturnFocusNode(state.nodeFocusedBeforeActivation)).then(finishDeactivation, finishDeactivation);
        return this;
      }
      finishDeactivation();
      return this;
    },
    pause: function pause2(pauseOptions) {
      if (state.paused || !state.active) {
        return this;
      }
      var onPause = getOption(pauseOptions, "onPause");
      var onPostPause = getOption(pauseOptions, "onPostPause");
      state.paused = true;
      onPause === null || onPause === void 0 || onPause();
      removeListeners();
      updateObservedNodes();
      onPostPause === null || onPostPause === void 0 || onPostPause();
      return this;
    },
    unpause: function unpause(unpauseOptions) {
      if (!state.paused || !state.active) {
        return this;
      }
      var onUnpause = getOption(unpauseOptions, "onUnpause");
      var onPostUnpause = getOption(unpauseOptions, "onPostUnpause");
      state.paused = false;
      onUnpause === null || onUnpause === void 0 || onUnpause();
      updateTabbableNodes();
      addListeners();
      updateObservedNodes();
      onPostUnpause === null || onPostUnpause === void 0 || onPostUnpause();
      return this;
    },
    updateContainerElements: function updateContainerElements(containerElements) {
      var elementsAsArray = [].concat(containerElements).filter(Boolean);
      state.containers = elementsAsArray.map(function(element2) {
        return typeof element2 === "string" ? doc.querySelector(element2) : element2;
      });
      if (state.active) {
        updateTabbableNodes();
      }
      updateObservedNodes();
      return this;
    }
  };
  trap.updateContainerElements(elements);
  return trap;
};
function createFocusTrap2(config = {}) {
  let trap;
  const { immediate, ...focusTrapOptions } = config;
  const hasFocus = writable(false);
  const isPaused = writable(false);
  const activate = (opts) => trap == null ? void 0 : trap.activate(opts);
  const deactivate = (opts) => {
    trap == null ? void 0 : trap.deactivate(opts);
  };
  const pause2 = () => {
    if (trap) {
      trap.pause();
      isPaused.set(true);
    }
  };
  const unpause = () => {
    if (trap) {
      trap.unpause();
      isPaused.set(false);
    }
  };
  const useFocusTrap = (node) => {
    trap = createFocusTrap$1(node, {
      ...focusTrapOptions,
      onActivate() {
        var _a;
        hasFocus.set(true);
        (_a = config.onActivate) == null ? void 0 : _a.call(config);
      },
      onDeactivate() {
        var _a;
        hasFocus.set(false);
        (_a = config.onDeactivate) == null ? void 0 : _a.call(config);
      }
    });
    if (immediate) {
      activate();
    }
    return {
      destroy() {
        deactivate();
        trap = void 0;
      }
    };
  };
  return {
    useFocusTrap,
    hasFocus: readonly(hasFocus),
    isPaused: readonly(isPaused),
    activate,
    deactivate,
    pause: pause2,
    unpause
  };
}
const defaultConfig = {
  floating: {},
  focusTrap: {},
  clickOutside: {},
  escapeKeydown: {},
  portal: "body"
};
const usePopper = (popperElement, args) => {
  popperElement.dataset.escapee = "";
  const { anchorElement, open, options } = args;
  if (!anchorElement || !open || !options) {
    return { destroy: noop };
  }
  const opts = { ...defaultConfig, ...options };
  const callbacks = [];
  if (opts.portal !== null) {
    const portal2 = usePortal(popperElement, opts.portal);
    if (portal2 == null ? void 0 : portal2.destroy) {
      callbacks.push(portal2.destroy);
    }
  }
  callbacks.push(useFloating(anchorElement, popperElement, opts.floating).destroy);
  if (opts.focusTrap !== null) {
    const { useFocusTrap } = createFocusTrap2({
      immediate: true,
      escapeDeactivates: false,
      allowOutsideClick: true,
      returnFocusOnDeactivate: false,
      fallbackFocus: popperElement,
      ...opts.focusTrap
    });
    const usedFocusTrap = useFocusTrap(popperElement);
    if (usedFocusTrap == null ? void 0 : usedFocusTrap.destroy) {
      callbacks.push(usedFocusTrap.destroy);
    }
  }
  if (opts.clickOutside !== null) {
    callbacks.push(useClickOutside(popperElement, {
      enabled: open,
      handler: (e) => {
        if (e.defaultPrevented)
          return;
        if (isHTMLElement$1(anchorElement) && !anchorElement.contains(e.target)) {
          open.set(false);
          anchorElement.focus();
        }
      },
      ...opts.clickOutside
    }).destroy);
  }
  if (opts.escapeKeydown !== null) {
    callbacks.push(useEscapeKeydown(popperElement, {
      enabled: open,
      handler: () => {
        open.set(false);
      },
      ...opts.escapeKeydown
    }).destroy);
  }
  const unsubscribe2 = executeCallbacks(...callbacks);
  return {
    destroy() {
      unsubscribe2();
    }
  };
};
const usePortal = (el, target = "body") => {
  let targetEl;
  if (!isHTMLElement$1(target) && typeof target !== "string") {
    return {
      destroy: noop
    };
  }
  async function update2(newTarget) {
    target = newTarget;
    if (typeof target === "string") {
      targetEl = document.querySelector(target);
      if (targetEl === null) {
        await tick();
        targetEl = document.querySelector(target);
      }
      if (targetEl === null) {
        throw new Error(`No element found matching css selector: "${target}"`);
      }
    } else if (target instanceof HTMLElement) {
      targetEl = target;
    } else {
      throw new TypeError(`Unknown portal target type: ${target === null ? "null" : typeof target}. Allowed types: string (CSS selector) or HTMLElement.`);
    }
    el.dataset.portal = "";
    targetEl.appendChild(el);
    el.hidden = false;
  }
  function destroy() {
    el.remove();
  }
  update2(target);
  return {
    update: update2,
    destroy
  };
};
function createLabel() {
  const root2 = builder("label", {
    action: (node) => {
      const mouseDown = addMeltEventListener(node, "mousedown", (e) => {
        if (!e.defaultPrevented && e.detail > 1) {
          e.preventDefault();
        }
      });
      return {
        destroy: mouseDown
      };
    }
  });
  return {
    elements: {
      root: root2
    }
  };
}
const INTERACTION_KEYS = [kbd.ARROW_LEFT, kbd.ESCAPE, kbd.ARROW_RIGHT, kbd.SHIFT, kbd.CAPS_LOCK, kbd.CONTROL, kbd.ALT, kbd.META, kbd.ENTER, kbd.F1, kbd.F2, kbd.F3, kbd.F4, kbd.F5, kbd.F6, kbd.F7, kbd.F8, kbd.F9, kbd.F10, kbd.F11, kbd.F12];
const defaults$1 = {
  positioning: {
    placement: "bottom",
    sameWidth: true
  },
  scrollAlignment: "nearest",
  loop: true,
  defaultOpen: false,
  closeOnOutsideClick: true,
  preventScroll: true,
  closeOnEscape: true,
  forceVisible: false,
  portal: void 0,
  builder: "listbox",
  disabled: false,
  required: false,
  name: void 0,
  typeahead: true,
  highlightOnHover: true,
  onOutsideClick: void 0
};
const listboxIdParts = ["trigger", "menu", "label"];
function createListbox(props) {
  const withDefaults = { ...defaults$1, ...props };
  const activeTrigger = writable(null);
  const highlightedItem = writable(null);
  const selectedWritable = withDefaults.selected ?? writable(withDefaults.defaultSelected);
  const selected = overridable(selectedWritable, withDefaults == null ? void 0 : withDefaults.onSelectedChange);
  const highlighted = derived(highlightedItem, ($highlightedItem) => $highlightedItem ? getOptionProps($highlightedItem) : void 0);
  const openWritable = withDefaults.open ?? writable(withDefaults.defaultOpen);
  const open = overridable(openWritable, withDefaults == null ? void 0 : withDefaults.onOpenChange);
  const options = toWritableStores({
    ...omit(withDefaults, "open", "defaultOpen", "builder", "ids"),
    multiple: withDefaults.multiple ?? false
  });
  const { scrollAlignment, loop: loop2, closeOnOutsideClick, closeOnEscape, preventScroll, portal: portal2, forceVisible, positioning, multiple, arrowSize, disabled, required, typeahead, name: nameProp, highlightOnHover, onOutsideClick } = options;
  const { name: name2, selector: selector2 } = createElHelpers(withDefaults.builder);
  const ids = toWritableStores({ ...generateIds(listboxIdParts), ...withDefaults.ids });
  const { handleTypeaheadSearch } = createTypeaheadSearch({
    onMatch: (element2) => {
      highlightedItem.set(element2);
      element2.scrollIntoView({ block: get_store_value(scrollAlignment) });
    },
    getCurrentItem() {
      return get_store_value(highlightedItem);
    }
  });
  function getOptionProps(el) {
    const value = el.getAttribute("data-value");
    const label3 = el.getAttribute("data-label");
    const disabled2 = el.hasAttribute("data-disabled");
    return {
      value: value ? JSON.parse(value) : value,
      label: label3 ?? el.textContent ?? void 0,
      disabled: disabled2 ? true : false
    };
  }
  const setOption = (newOption) => {
    selected.update(($option) => {
      const $multiple = get_store_value(multiple);
      if ($multiple) {
        const optionArr = Array.isArray($option) ? $option : [];
        return toggle(newOption, optionArr, (itemA, itemB) => dequal(itemA.value, itemB.value));
      }
      return newOption;
    });
  };
  function selectItem(item) {
    const props2 = getOptionProps(item);
    setOption(props2);
  }
  async function openMenu() {
    open.set(true);
    const triggerEl = document.getElementById(get_store_value(ids.trigger));
    if (!triggerEl)
      return;
    activeTrigger.set(triggerEl);
    await tick();
    const menuElement = document.getElementById(get_store_value(ids.menu));
    if (!isHTMLElement$1(menuElement))
      return;
    const selectedItem = menuElement.querySelector("[aria-selected=true]");
    if (!isHTMLElement$1(selectedItem))
      return;
    highlightedItem.set(selectedItem);
  }
  function closeMenu() {
    open.set(false);
    highlightedItem.set(null);
  }
  const isVisible = derivedVisible({ open, forceVisible, activeTrigger });
  const isSelected = derived([selected], ([$selected]) => {
    return (value) => {
      if (Array.isArray($selected)) {
        return $selected.some((o) => dequal(o.value, value));
      }
      if (isObject(value)) {
        return dequal($selected == null ? void 0 : $selected.value, stripValues(value, void 0));
      }
      return dequal($selected == null ? void 0 : $selected.value, value);
    };
  });
  const isHighlighted = derived([highlighted], ([$value]) => {
    return (item) => {
      return dequal($value == null ? void 0 : $value.value, item);
    };
  });
  const trigger = builder(name2("trigger"), {
    stores: [open, highlightedItem, disabled, ids.menu, ids.trigger, ids.label],
    returned: ([$open, $highlightedItem, $disabled, $menuId, $triggerId, $labelId]) => {
      return {
        "aria-activedescendant": $highlightedItem == null ? void 0 : $highlightedItem.id,
        "aria-autocomplete": "list",
        "aria-controls": $menuId,
        "aria-expanded": $open,
        "aria-labelledby": $labelId,
        // autocomplete: 'off',
        id: $triggerId,
        role: "combobox",
        disabled: disabledAttr($disabled)
      };
    },
    action: (node) => {
      const isInput3 = isHTMLInputElement(node);
      const unsubscribe2 = executeCallbacks(
        addMeltEventListener(node, "click", () => {
          node.focus();
          const $open = get_store_value(open);
          if ($open) {
            closeMenu();
          } else {
            openMenu();
          }
        }),
        // Handle all input key events including typing, meta, and navigation.
        addMeltEventListener(node, "keydown", (e) => {
          const $open = get_store_value(open);
          if (!$open) {
            if (INTERACTION_KEYS.includes(e.key)) {
              return;
            }
            if (e.key === kbd.TAB) {
              return;
            }
            if (e.key === kbd.BACKSPACE && isInput3 && node.value === "") {
              return;
            }
            if (e.key === kbd.SPACE && isHTMLButtonElement(node)) {
              return;
            }
            openMenu();
            tick().then(() => {
              const $selectedItem = get_store_value(selected);
              if ($selectedItem)
                return;
              const menuEl = document.getElementById(get_store_value(ids.menu));
              if (!isHTMLElement$1(menuEl))
                return;
              const enabledItems = Array.from(menuEl.querySelectorAll(`${selector2("item")}:not([data-disabled]):not([data-hidden])`)).filter((item) => isHTMLElement$1(item));
              if (!enabledItems.length)
                return;
              if (e.key === kbd.ARROW_DOWN) {
                highlightedItem.set(enabledItems[0]);
                enabledItems[0].scrollIntoView({ block: get_store_value(scrollAlignment) });
              } else if (e.key === kbd.ARROW_UP) {
                highlightedItem.set(last(enabledItems));
                last(enabledItems).scrollIntoView({ block: get_store_value(scrollAlignment) });
              }
            });
          }
          if (e.key === kbd.TAB) {
            closeMenu();
            return;
          }
          if (e.key === kbd.ENTER || e.key === kbd.SPACE && isHTMLButtonElement(node)) {
            e.preventDefault();
            const $highlightedItem = get_store_value(highlightedItem);
            if ($highlightedItem) {
              selectItem($highlightedItem);
            }
            if (!get_store_value(multiple)) {
              closeMenu();
            }
          }
          if (e.key === kbd.ARROW_UP && e.altKey) {
            closeMenu();
          }
          if (FIRST_LAST_KEYS.includes(e.key)) {
            e.preventDefault();
            const menuElement = document.getElementById(get_store_value(ids.menu));
            if (!isHTMLElement$1(menuElement))
              return;
            const itemElements = getOptions(menuElement);
            if (!itemElements.length)
              return;
            const candidateNodes = itemElements.filter((opt) => !isElementDisabled(opt) && opt.dataset.hidden === void 0);
            const $currentItem = get_store_value(highlightedItem);
            const currentIndex = $currentItem ? candidateNodes.indexOf($currentItem) : -1;
            const $loop = get_store_value(loop2);
            const $scrollAlignment = get_store_value(scrollAlignment);
            let nextItem;
            switch (e.key) {
              case kbd.ARROW_DOWN:
                nextItem = next(candidateNodes, currentIndex, $loop);
                break;
              case kbd.ARROW_UP:
                nextItem = prev(candidateNodes, currentIndex, $loop);
                break;
              case kbd.PAGE_DOWN:
                nextItem = forward(candidateNodes, currentIndex, 10, $loop);
                break;
              case kbd.PAGE_UP:
                nextItem = back(candidateNodes, currentIndex, 10, $loop);
                break;
              case kbd.HOME:
                nextItem = candidateNodes[0];
                break;
              case kbd.END:
                nextItem = last(candidateNodes);
                break;
              default:
                return;
            }
            highlightedItem.set(nextItem);
            nextItem == null ? void 0 : nextItem.scrollIntoView({ block: $scrollAlignment });
          } else if (get_store_value(typeahead)) {
            const menuEl = document.getElementById(get_store_value(ids.menu));
            if (!isHTMLElement$1(menuEl))
              return;
            handleTypeaheadSearch(e.key, getOptions(menuEl));
          }
        })
      );
      let unsubEscapeKeydown = noop;
      const escape = useEscapeKeydown(node, {
        handler: closeMenu,
        enabled: derived([open, closeOnEscape], ([$open, $closeOnEscape]) => {
          return $open && $closeOnEscape;
        })
      });
      if (escape && escape.destroy) {
        unsubEscapeKeydown = escape.destroy;
      }
      return {
        destroy() {
          unsubscribe2();
          unsubEscapeKeydown();
        }
      };
    }
  });
  const menu2 = builder(name2("menu"), {
    stores: [isVisible, ids.menu],
    returned: ([$isVisible, $menuId]) => {
      return {
        hidden: $isVisible ? void 0 : true,
        id: $menuId,
        role: "listbox",
        style: styleToString({ display: $isVisible ? void 0 : "none" })
      };
    },
    action: (node) => {
      let unsubPopper = noop;
      const unsubscribe2 = executeCallbacks(
        // Bind the popper portal to the input element.
        effect([isVisible, portal2, closeOnOutsideClick, positioning, activeTrigger], ([$isVisible, $portal, $closeOnOutsideClick, $positioning, $activeTrigger]) => {
          unsubPopper();
          if (!$isVisible || !$activeTrigger)
            return;
          const ignoreHandler = createClickOutsideIgnore(get_store_value(ids.trigger));
          const popper = usePopper(node, {
            anchorElement: $activeTrigger,
            open,
            options: {
              floating: $positioning,
              focusTrap: null,
              clickOutside: $closeOnOutsideClick ? {
                handler: (e) => {
                  var _a;
                  (_a = get_store_value(onOutsideClick)) == null ? void 0 : _a(e);
                  if (e.defaultPrevented)
                    return;
                  const target = e.target;
                  if (!isElement$1(target))
                    return;
                  if (target === $activeTrigger || $activeTrigger.contains(target)) {
                    return;
                  }
                  closeMenu();
                },
                ignore: ignoreHandler
              } : null,
              escapeKeydown: null,
              portal: getPortalDestination(node, $portal)
            }
          });
          if (popper && popper.destroy) {
            unsubPopper = popper.destroy;
          }
        })
      );
      return {
        destroy: () => {
          unsubscribe2();
          unsubPopper();
        }
      };
    }
  });
  const { elements: { root: labelBuilder } } = createLabel();
  const { action: labelAction } = get_store_value(labelBuilder);
  const label2 = builder(name2("label"), {
    stores: [ids.label, ids.trigger],
    returned: ([$labelId, $triggerId]) => {
      return {
        id: $labelId,
        for: $triggerId
      };
    },
    action: labelAction
  });
  const option = builder(name2("option"), {
    stores: [isSelected],
    returned: ([$isSelected]) => (props2) => {
      const selected2 = $isSelected(props2.value);
      return {
        "data-value": JSON.stringify(props2.value),
        "data-label": props2.label,
        "data-disabled": disabledAttr(props2.disabled),
        "aria-disabled": props2.disabled ? true : void 0,
        "aria-selected": selected2,
        "data-selected": selected2 ? "" : void 0,
        id: generateId(),
        role: "option"
      };
    },
    action: (node) => {
      const unsubscribe2 = executeCallbacks(addMeltEventListener(node, "click", (e) => {
        if (isElementDisabled(node)) {
          e.preventDefault();
          return;
        }
        selectItem(node);
        if (!get_store_value(multiple)) {
          closeMenu();
        }
      }), effect(highlightOnHover, ($highlightOnHover) => {
        if (!$highlightOnHover)
          return;
        const unsub = executeCallbacks(addMeltEventListener(node, "mouseover", () => {
          highlightedItem.set(node);
        }), addMeltEventListener(node, "mouseleave", () => {
          highlightedItem.set(null);
        }));
        return unsub;
      }));
      return { destroy: unsubscribe2 };
    }
  });
  const group2 = builder(name2("group"), {
    returned: () => {
      return (groupId) => ({
        role: "group",
        "aria-labelledby": groupId
      });
    }
  });
  const groupLabel = builder(name2("group-label"), {
    returned: () => {
      return (groupId) => ({
        id: groupId
      });
    }
  });
  const hiddenInput = builder(name2("hidden-input"), {
    stores: [selected, required, nameProp],
    returned: ([$selected, $required, $name]) => {
      const value = Array.isArray($selected) ? $selected.map((o) => o.value) : $selected == null ? void 0 : $selected.value;
      return {
        ...hiddenInputAttrs,
        required: $required ? true : void 0,
        value,
        name: $name
      };
    }
  });
  const arrow2 = builder(name2("arrow"), {
    stores: arrowSize,
    returned: ($arrowSize) => ({
      "data-arrow": true,
      style: styleToString({
        position: "absolute",
        width: `var(--arrow-size, ${$arrowSize}px)`,
        height: `var(--arrow-size, ${$arrowSize}px)`
      })
    })
  });
  safeOnMount(() => {
    if (!isBrowser)
      return;
    const menuEl = document.getElementById(get_store_value(ids.menu));
    if (!menuEl)
      return;
    const triggerEl = document.getElementById(get_store_value(ids.trigger));
    if (triggerEl) {
      activeTrigger.set(triggerEl);
    }
    const selectedEl = menuEl.querySelector("[data-selected]");
    if (!isHTMLElement$1(selectedEl))
      return;
  });
  effect([highlightedItem], ([$highlightedItem]) => {
    if (!isBrowser)
      return;
    const menuElement = document.getElementById(get_store_value(ids.menu));
    if (!isHTMLElement$1(menuElement))
      return;
    getOptions(menuElement).forEach((node) => {
      if (node === $highlightedItem) {
        addHighlight(node);
      } else {
        removeHighlight(node);
      }
    });
  });
  effect([open], ([$open]) => {
    if (!isBrowser)
      return;
    let unsubScroll = noop;
    if (get_store_value(preventScroll) && $open) {
      unsubScroll = removeScroll();
    }
    return () => {
      unsubScroll();
    };
  });
  return {
    ids,
    elements: {
      trigger,
      group: group2,
      option,
      menu: menu2,
      groupLabel,
      label: label2,
      hiddenInput,
      arrow: arrow2
    },
    states: {
      open,
      selected,
      highlighted,
      highlightedItem
    },
    helpers: {
      isSelected,
      isHighlighted,
      closeMenu
    },
    options
  };
}
const { name: name$1 } = createElHelpers("combobox");
function createCombobox(props) {
  const listbox = createListbox({ ...props, builder: "combobox", typeahead: false });
  const inputValue = writable("");
  const touchedInput = writable(false);
  const input = builder(name$1("input"), {
    stores: [listbox.elements.trigger, inputValue],
    returned: ([$trigger, $inputValue]) => {
      return {
        ...omit($trigger, "action"),
        role: "combobox",
        value: $inputValue
      };
    },
    action: (node) => {
      const unsubscribe2 = executeCallbacks(
        addMeltEventListener(node, "input", (e) => {
          if (!isHTMLInputElement(e.target) && !isContentEditable$1(e.target))
            return;
          touchedInput.set(true);
        }),
        // This shouldn't be cancelled ever, so we don't use addMeltEventListener.
        addEventListener(node, "input", (e) => {
          if (isHTMLInputElement(e.target)) {
            inputValue.set(e.target.value);
          }
          if (isContentEditable$1(e.target)) {
            inputValue.set(e.target.innerText);
          }
        })
      );
      let unsubEscapeKeydown = noop;
      const escape = useEscapeKeydown(node, {
        handler: () => {
          listbox.helpers.closeMenu();
        }
      });
      if (escape && escape.destroy) {
        unsubEscapeKeydown = escape.destroy;
      }
      const { destroy } = listbox.elements.trigger(node);
      return {
        destroy() {
          destroy == null ? void 0 : destroy();
          unsubscribe2();
          unsubEscapeKeydown();
        }
      };
    }
  });
  effect(listbox.states.open, ($open) => {
    if (!$open) {
      touchedInput.set(false);
    }
  });
  return {
    ...listbox,
    elements: {
      ...omit(listbox.elements, "trigger"),
      input
    },
    states: {
      ...listbox.states,
      touchedInput,
      inputValue
    }
  };
}
const defaults = {
  defaultValue: [],
  min: 0,
  max: 100,
  step: 1,
  orientation: "horizontal",
  dir: "ltr",
  disabled: false
};
const { name } = createElHelpers("slider");
const createSlider = (props) => {
  const withDefaults = { ...defaults, ...props };
  const options = toWritableStores(omit(withDefaults, "value", "onValueChange", "defaultValue"));
  const { min: min2, max: max2, step, orientation, dir, disabled } = options;
  const valueWritable = withDefaults.value ?? writable(withDefaults.defaultValue);
  const value = overridable(valueWritable, withDefaults == null ? void 0 : withDefaults.onValueChange);
  const isActive = writable(false);
  const currentThumbIndex = writable(0);
  const activeThumb = writable(null);
  const meltIds = generateIds(["root"]);
  const updatePosition = (val, index) => {
    value.update((prev2) => {
      if (!prev2)
        return [val];
      if (prev2[index] === val)
        return prev2;
      const newValue = [...prev2];
      const direction2 = newValue[index] > val ? -1 : 1;
      function swap() {
        newValue[index] = newValue[index + direction2];
        newValue[index + direction2] = val;
        const thumbs2 = getAllThumbs();
        if (thumbs2) {
          thumbs2[index + direction2].focus();
          activeThumb.set({ thumb: thumbs2[index + direction2], index: index + direction2 });
        }
      }
      if (direction2 === -1 && val < newValue[index - 1]) {
        swap();
        return newValue;
      } else if (direction2 === 1 && val > newValue[index + 1]) {
        swap();
        return newValue;
      }
      const $min = get_store_value(min2);
      const $max = get_store_value(max2);
      const $step = get_store_value(step);
      newValue[index] = snapValueToStep(val, $min, $max, $step);
      return newValue;
    });
  };
  const getAllThumbs = () => {
    const root3 = getElementByMeltId(meltIds.root);
    if (!root3)
      return null;
    return Array.from(root3.querySelectorAll('[data-melt-part="thumb"]')).filter((thumb) => isHTMLElement$1(thumb));
  };
  const position = derived([min2, max2], ([$min, $max]) => {
    return (val) => {
      const pos = (val - $min) / ($max - $min) * 100;
      return pos;
    };
  });
  const direction = derived([orientation, dir], ([$orientation, $dir]) => {
    if ($orientation === "horizontal") {
      return $dir === "rtl" ? "rl" : "lr";
    } else {
      return $dir === "rtl" ? "tb" : "bt";
    }
  });
  const root2 = builder(name(), {
    stores: [disabled, orientation, dir],
    returned: ([$disabled, $orientation, $dir]) => {
      return {
        dir: $dir,
        disabled: disabledAttr($disabled),
        "data-disabled": disabledAttr($disabled),
        "data-orientation": $orientation,
        style: $disabled ? void 0 : `touch-action: ${$orientation === "horizontal" ? "pan-y" : "pan-x"}`,
        "data-melt-id": meltIds.root
      };
    }
  });
  const range2 = builder(name("range"), {
    stores: [value, direction, position],
    returned: ([$value, $direction, $position]) => {
      const minimum = $value.length > 1 ? $position(Math.min(...$value) ?? 0) : 0;
      const maximum = 100 - $position(Math.max(...$value) ?? 0);
      const style2 = {
        position: "absolute"
      };
      switch ($direction) {
        case "lr": {
          style2.left = `${minimum}%`;
          style2.right = `${maximum}%`;
          break;
        }
        case "rl": {
          style2.right = `${minimum}%`;
          style2.left = `${maximum}%`;
          break;
        }
        case "bt": {
          style2.bottom = `${minimum}%`;
          style2.top = `${maximum}%`;
          break;
        }
        case "tb": {
          style2.top = `${minimum}%`;
          style2.bottom = `${maximum}%`;
          break;
        }
      }
      return {
        style: styleToString(style2)
      };
    }
  });
  const thumbs = builderArray(name("thumb"), {
    stores: [value, position, min2, max2, disabled, orientation, direction],
    returned: ([$value, $position, $min, $max, $disabled, $orientation, $direction]) => {
      return Array.from({ length: $value.length || 1 }, (_, i2) => {
        const currentThumb = get_store_value(currentThumbIndex);
        if (currentThumb < $value.length) {
          currentThumbIndex.update((prev2) => prev2 + 1);
        }
        const thumbValue = $value[i2];
        const thumbPosition = `${$position(thumbValue)}%`;
        const style2 = {
          position: "absolute"
        };
        switch ($direction) {
          case "lr": {
            style2.left = thumbPosition;
            style2.translate = "-50% 0";
            break;
          }
          case "rl": {
            style2.right = thumbPosition;
            style2.translate = "50% 0";
            break;
          }
          case "bt": {
            style2.bottom = thumbPosition;
            style2.translate = "0 50%";
            break;
          }
          case "tb": {
            style2.top = thumbPosition;
            style2.translate = "0 -50%";
            break;
          }
        }
        return {
          role: "slider",
          "aria-valuemin": $min,
          "aria-valuemax": $max,
          "aria-valuenow": thumbValue,
          "aria-disabled": disabledAttr($disabled),
          "aria-orientation": $orientation,
          "data-melt-part": "thumb",
          "data-value": thumbValue,
          style: styleToString(style2),
          tabindex: $disabled ? -1 : 0
        };
      });
    },
    action: (node) => {
      const unsub = addMeltEventListener(node, "keydown", (event) => {
        if (get_store_value(disabled))
          return;
        const target = event.currentTarget;
        if (!isHTMLElement$1(target))
          return;
        const thumbs2 = getAllThumbs();
        if (!(thumbs2 == null ? void 0 : thumbs2.length))
          return;
        const index = thumbs2.indexOf(target);
        currentThumbIndex.set(index);
        if (![
          kbd.ARROW_LEFT,
          kbd.ARROW_RIGHT,
          kbd.ARROW_UP,
          kbd.ARROW_DOWN,
          kbd.HOME,
          kbd.END
        ].includes(event.key)) {
          return;
        }
        event.preventDefault();
        const $min = get_store_value(min2);
        const $max = get_store_value(max2);
        const $step = get_store_value(step);
        const $value = get_store_value(value);
        const $orientation = get_store_value(orientation);
        const $direction = get_store_value(direction);
        const thumbValue = $value[index];
        switch (event.key) {
          case kbd.HOME: {
            updatePosition($min, index);
            break;
          }
          case kbd.END: {
            updatePosition($max, index);
            break;
          }
          case kbd.ARROW_LEFT: {
            if ($orientation !== "horizontal")
              break;
            if (event.metaKey) {
              const newValue = $direction === "rl" ? $max : $min;
              updatePosition(newValue, index);
            } else if ($direction === "rl" && thumbValue < $max) {
              updatePosition(thumbValue + $step, index);
            } else if ($direction === "lr" && thumbValue > $min) {
              updatePosition(thumbValue - $step, index);
            }
            break;
          }
          case kbd.ARROW_RIGHT: {
            if ($orientation !== "horizontal")
              break;
            if (event.metaKey) {
              const newValue = $direction === "rl" ? $min : $max;
              updatePosition(newValue, index);
            } else if ($direction === "rl" && thumbValue > $min) {
              updatePosition(thumbValue - $step, index);
            } else if ($direction === "lr" && thumbValue < $max) {
              updatePosition(thumbValue + $step, index);
            }
            break;
          }
          case kbd.ARROW_UP: {
            if (event.metaKey) {
              const newValue = $direction === "tb" ? $min : $max;
              updatePosition(newValue, index);
            } else if ($direction === "tb" && thumbValue > $min) {
              updatePosition(thumbValue - $step, index);
            } else if ($direction !== "tb" && thumbValue < $max) {
              updatePosition(thumbValue + $step, index);
            }
            break;
          }
          case kbd.ARROW_DOWN: {
            if (event.metaKey) {
              const newValue = $direction === "tb" ? $max : $min;
              updatePosition(newValue, index);
            } else if ($direction === "tb" && thumbValue < $max) {
              updatePosition(thumbValue + $step, index);
            } else if ($direction !== "tb" && thumbValue > $min) {
              updatePosition(thumbValue - $step, index);
            }
            break;
          }
        }
      });
      return {
        destroy: unsub
      };
    }
  });
  const ticks2 = builderArray(name("tick"), {
    stores: [value, min2, max2, step, direction],
    returned: ([$value, $min, $max, $step, $direction]) => {
      const difference = $max - $min;
      let count = Math.ceil(difference / $step);
      if (difference % $step == 0) {
        count++;
      }
      return Array.from({ length: count }, (_, i2) => {
        const tickPosition = `${i2 * ($step / ($max - $min)) * 100}%`;
        const isFirst = i2 === 0;
        const isLast = i2 === count - 1;
        const offsetPercentage = isFirst ? 0 : isLast ? -100 : -50;
        const style2 = {
          position: "absolute"
        };
        switch ($direction) {
          case "lr": {
            style2.left = tickPosition;
            style2.translate = `${offsetPercentage}% 0`;
            break;
          }
          case "rl": {
            style2.right = tickPosition;
            style2.translate = `${-offsetPercentage}% 0`;
            break;
          }
          case "bt": {
            style2.bottom = tickPosition;
            style2.translate = `0 ${-offsetPercentage}%`;
            break;
          }
          case "tb": {
            style2.top = tickPosition;
            style2.translate = `0 ${offsetPercentage}%`;
            break;
          }
        }
        const tickValue = $min + i2 * $step;
        const bounded = $value.length === 1 ? tickValue <= $value[0] : $value[0] <= tickValue && tickValue <= $value[$value.length - 1];
        return {
          "data-bounded": bounded ? true : void 0,
          "data-value": tickValue,
          style: styleToString(style2)
        };
      });
    }
  });
  effect([root2, min2, max2, disabled, orientation, direction, step], ([$root, $min, $max, $disabled, $orientation, $direction, $step]) => {
    if (!isBrowser || $disabled)
      return;
    const applyPosition = (clientXY, activeThumbIdx, start2, end) => {
      const percent = (clientXY - start2) / (end - start2);
      const val = percent * ($max - $min) + $min;
      if (val < $min) {
        updatePosition($min, activeThumbIdx);
      } else if (val > $max) {
        updatePosition($max, activeThumbIdx);
      } else {
        const step2 = $step;
        const min3 = $min;
        const currentStep = Math.floor((val - min3) / step2);
        const midpointOfCurrentStep = min3 + currentStep * step2 + step2 / 2;
        const midpointOfNextStep = min3 + (currentStep + 1) * step2 + step2 / 2;
        const newValue = val >= midpointOfCurrentStep && val < midpointOfNextStep ? (currentStep + 1) * step2 + min3 : currentStep * step2 + min3;
        if (newValue <= $max) {
          updatePosition(newValue, activeThumbIdx);
        }
      }
    };
    const getClosestThumb = (e) => {
      const thumbs2 = getAllThumbs();
      if (!thumbs2)
        return;
      thumbs2.forEach((thumb2) => thumb2.blur());
      const distances = thumbs2.map((thumb2) => {
        if ($orientation === "horizontal") {
          const { left: left2, right: right2 } = thumb2.getBoundingClientRect();
          return Math.abs(e.clientX - (left2 + right2) / 2);
        } else {
          const { top: top2, bottom: bottom2 } = thumb2.getBoundingClientRect();
          return Math.abs(e.clientY - (top2 + bottom2) / 2);
        }
      });
      const thumb = thumbs2[distances.indexOf(Math.min(...distances))];
      const index = thumbs2.indexOf(thumb);
      return { thumb, index };
    };
    const pointerMove = (e) => {
      if (!get_store_value(isActive))
        return;
      e.preventDefault();
      e.stopPropagation();
      const sliderEl = getElementByMeltId($root["data-melt-id"]);
      const closestThumb = get_store_value(activeThumb);
      if (!sliderEl || !closestThumb)
        return;
      closestThumb.thumb.focus();
      const { left: left2, right: right2, top: top2, bottom: bottom2 } = sliderEl.getBoundingClientRect();
      switch ($direction) {
        case "lr": {
          applyPosition(e.clientX, closestThumb.index, left2, right2);
          break;
        }
        case "rl": {
          applyPosition(e.clientX, closestThumb.index, right2, left2);
          break;
        }
        case "bt": {
          applyPosition(e.clientY, closestThumb.index, bottom2, top2);
          break;
        }
        case "tb": {
          applyPosition(e.clientY, closestThumb.index, top2, bottom2);
          break;
        }
      }
    };
    const pointerDown = (e) => {
      if (e.button !== 0)
        return;
      const sliderEl = getElementByMeltId($root["data-melt-id"]);
      const closestThumb = getClosestThumb(e);
      if (!closestThumb || !sliderEl)
        return;
      const target = e.target;
      if (!isHTMLElement$1(target) || !sliderEl.contains(target)) {
        return;
      }
      e.preventDefault();
      activeThumb.set(closestThumb);
      closestThumb.thumb.focus();
      isActive.set(true);
      pointerMove(e);
    };
    const pointerUp = () => {
      isActive.set(false);
    };
    const unsub = executeCallbacks(addEventListener(document, "pointerdown", pointerDown), addEventListener(document, "pointerup", pointerUp), addEventListener(document, "pointerleave", pointerUp), addEventListener(document, "pointermove", pointerMove));
    return () => {
      unsub();
    };
  });
  effect([step, min2, max2, value], function fixValue([$step, $min, $max, $value]) {
    const isValidValue = (v) => {
      const snappedValue = snapValueToStep(v, $min, $max, $step);
      return snappedValue === v;
    };
    const gcv = (v) => {
      return snapValueToStep(v, $min, $max, $step);
    };
    if ($value.some((v) => !isValidValue(v))) {
      value.update((prev2) => {
        return prev2.map(gcv);
      });
    }
  });
  return {
    elements: {
      root: root2,
      thumbs,
      range: range2,
      ticks: ticks2
    },
    states: {
      value
    },
    options
  };
};
function create_if_block$6(ctx) {
  let p;
  let t;
  return {
    c() {
      p = element("p");
      t = text(
        /*name*/
        ctx[1]
      );
    },
    m(target, anchor2) {
      insert(target, p, anchor2);
      append$1(p, t);
    },
    p(ctx2, dirty) {
      if (dirty & /*name*/
      2)
        set_data(
          t,
          /*name*/
          ctx2[1]
        );
    },
    d(detaching) {
      if (detaching) {
        detach(p);
      }
    }
  };
}
function create_fragment$e(ctx) {
  let div1;
  let div0;
  let t0;
  let p;
  let span0;
  let t1_value = (
    /*valueDisplay*/
    ctx[2](
      /*value*/
      ctx[0]
    ) + ""
  );
  let t1;
  let t2;
  let span4;
  let span2;
  let span1;
  let t3;
  let span3;
  let mounted;
  let dispose;
  let if_block = (
    /*name*/
    ctx[1] != "" && create_if_block$6(ctx)
  );
  let p_levels = [
    /*$labelRoot*/
    ctx[4],
    {
      class: "mb-0.5 font-medium text-magnum-900"
    },
    { "data-melt-part": "root" }
  ];
  let p_data = {};
  for (let i2 = 0; i2 < p_levels.length; i2 += 1) {
    p_data = assign(p_data, p_levels[i2]);
  }
  let span1_levels = [
    /*$range*/
    ctx[6],
    { class: "h-[3px] bg-black" }
  ];
  let span_data_1 = {};
  for (let i2 = 0; i2 < span1_levels.length; i2 += 1) {
    span_data_1 = assign(span_data_1, span1_levels[i2]);
  }
  let span3_levels = [
    /*__MELTUI_BUILDER_0__*/
    ctx[3],
    {
      class: "h-5 w-5 rounded-full bg-black focus:ring-4 focus:!ring-black/40"
    }
  ];
  let span_data = {};
  for (let i2 = 0; i2 < span3_levels.length; i2 += 1) {
    span_data = assign(span_data, span3_levels[i2]);
  }
  let span4_levels = [
    /*$root*/
    ctx[5],
    {
      class: "p-3 relative flex h-[20px] w-full items-center"
    }
  ];
  let span_data_3 = {};
  for (let i2 = 0; i2 < span4_levels.length; i2 += 1) {
    span_data_3 = assign(span_data_3, span4_levels[i2]);
  }
  return {
    c() {
      div1 = element("div");
      div0 = element("div");
      if (if_block)
        if_block.c();
      t0 = space();
      p = element("p");
      span0 = element("span");
      t1 = text(t1_value);
      t2 = space();
      span4 = element("span");
      span2 = element("span");
      span1 = element("span");
      t3 = space();
      span3 = element("span");
      attr(span0, "class", "relative top-1");
      set_attributes(p, p_data);
      set_attributes(span1, span_data_1);
      attr(span2, "class", "pl-3 pr-3 h-[3px] w-full bg-black/40");
      set_attributes(span3, span_data);
      set_attributes(span4, span_data_3);
      attr(div0, "class", "flex flex-col items-start justify-center");
      attr(div1, "class", "p-3 shadow-sm");
    },
    m(target, anchor2) {
      insert(target, div1, anchor2);
      append$1(div1, div0);
      if (if_block)
        if_block.m(div0, null);
      append$1(div0, t0);
      append$1(div0, p);
      append$1(p, span0);
      append$1(span0, t1);
      append$1(div0, t2);
      append$1(div0, span4);
      append$1(span4, span2);
      append$1(span2, span1);
      append$1(span4, t3);
      append$1(span4, span3);
      if (!mounted) {
        dispose = [
          action_destroyer(
            /*$labelRoot*/
            ctx[4].action(p)
          ),
          action_destroyer(
            /*$range*/
            ctx[6].action(span1)
          ),
          action_destroyer(
            /*__MELTUI_BUILDER_0__*/
            ctx[3].action(span3)
          ),
          action_destroyer(
            /*$root*/
            ctx[5].action(span4)
          )
        ];
        mounted = true;
      }
    },
    p(ctx2, [dirty]) {
      if (
        /*name*/
        ctx2[1] != ""
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
        } else {
          if_block = create_if_block$6(ctx2);
          if_block.c();
          if_block.m(div0, t0);
        }
      } else if (if_block) {
        if_block.d(1);
        if_block = null;
      }
      if (dirty & /*valueDisplay, value*/
      5 && t1_value !== (t1_value = /*valueDisplay*/
      ctx2[2](
        /*value*/
        ctx2[0]
      ) + ""))
        set_data(t1, t1_value);
      set_attributes(p, p_data = get_spread_update(p_levels, [
        dirty & /*$labelRoot*/
        16 && /*$labelRoot*/
        ctx2[4],
        {
          class: "mb-0.5 font-medium text-magnum-900"
        },
        { "data-melt-part": "root" }
      ]));
      set_attributes(span1, span_data_1 = get_spread_update(span1_levels, [dirty & /*$range*/
      64 && /*$range*/
      ctx2[6], { class: "h-[3px] bg-black" }]));
      set_attributes(span3, span_data = get_spread_update(span3_levels, [
        dirty & /*__MELTUI_BUILDER_0__*/
        8 && /*__MELTUI_BUILDER_0__*/
        ctx2[3],
        {
          class: "h-5 w-5 rounded-full bg-black focus:ring-4 focus:!ring-black/40"
        }
      ]));
      set_attributes(span4, span_data_3 = get_spread_update(span4_levels, [
        dirty & /*$root*/
        32 && /*$root*/
        ctx2[5],
        {
          class: "p-3 relative flex h-[20px] w-full items-center"
        }
      ]));
    },
    i: noop$2,
    o: noop$2,
    d(detaching) {
      if (detaching) {
        detach(div1);
      }
      if (if_block)
        if_block.d();
      mounted = false;
      run_all(dispose);
    }
  };
}
function instance$d($$self, $$props, $$invalidate) {
  let __MELTUI_BUILDER_0__;
  let $thumbs;
  let $labelRoot;
  let $root;
  let $range;
  let { name: name2 = "" } = $$props;
  let { min: min2 = 0 } = $$props;
  let { max: max2 = 100 } = $$props;
  let { step = 1 } = $$props;
  let { value = 10 } = $$props;
  let { valueDisplay = (v) => v.toString() } = $$props;
  const valueWritableArray = writable([value]);
  valueWritableArray.subscribe((v) => $$invalidate(0, value = v[0]));
  const { elements: { root: root2, range: range2, thumbs } } = createSlider({
    min: min2,
    max: max2,
    step,
    dir: "ltr",
    value: valueWritableArray
  });
  component_subscribe($$self, root2, (value2) => $$invalidate(5, $root = value2));
  component_subscribe($$self, range2, (value2) => $$invalidate(6, $range = value2));
  component_subscribe($$self, thumbs, (value2) => $$invalidate(14, $thumbs = value2));
  const label2 = createLabel();
  const labelRoot = label2.elements.root;
  component_subscribe($$self, labelRoot, (value2) => $$invalidate(4, $labelRoot = value2));
  $$self.$$set = ($$props2) => {
    if ("name" in $$props2)
      $$invalidate(1, name2 = $$props2.name);
    if ("min" in $$props2)
      $$invalidate(11, min2 = $$props2.min);
    if ("max" in $$props2)
      $$invalidate(12, max2 = $$props2.max);
    if ("step" in $$props2)
      $$invalidate(13, step = $$props2.step);
    if ("value" in $$props2)
      $$invalidate(0, value = $$props2.value);
    if ("valueDisplay" in $$props2)
      $$invalidate(2, valueDisplay = $$props2.valueDisplay);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty & /*$thumbs*/
    16384) {
      $$invalidate(3, __MELTUI_BUILDER_0__ = $thumbs[0]);
    }
  };
  return [
    value,
    name2,
    valueDisplay,
    __MELTUI_BUILDER_0__,
    $labelRoot,
    $root,
    $range,
    root2,
    range2,
    thumbs,
    labelRoot,
    min2,
    max2,
    step,
    $thumbs
  ];
}
class Slider extends SvelteComponent {
  constructor(options) {
    super();
    init$1(this, options, instance$d, create_fragment$e, safe_not_equal, {
      name: 1,
      min: 11,
      max: 12,
      step: 13,
      value: 0,
      valueDisplay: 2
    });
  }
}
function create_if_block_1$2(ctx) {
  let g;
  let g_transform_value;
  return {
    c() {
      g = svg_element("g");
      attr(g, "transform", g_transform_value = `translate(0,${/*height*/
      ctx[2]})`);
    },
    m(target, anchor2) {
      insert(target, g, anchor2);
      ctx[15](g);
    },
    p(ctx2, dirty) {
      if (dirty & /*height*/
      4 && g_transform_value !== (g_transform_value = `translate(0,${/*height*/
      ctx2[2]})`)) {
        attr(g, "transform", g_transform_value);
      }
    },
    d(detaching) {
      if (detaching) {
        detach(g);
      }
      ctx[15](null);
    }
  };
}
function create_if_block$5(ctx) {
  let g;
  return {
    c() {
      g = svg_element("g");
    },
    m(target, anchor2) {
      insert(target, g, anchor2);
      ctx[16](g);
    },
    p: noop$2,
    d(detaching) {
      if (detaching) {
        detach(g);
      }
      ctx[16](null);
    }
  };
}
function create_fragment$d(ctx) {
  let g;
  let if_block0_anchor;
  let if_block0 = (
    /*showAxisX*/
    ctx[0] && create_if_block_1$2(ctx)
  );
  let if_block1 = (
    /*showAxisY*/
    ctx[1] && create_if_block$5(ctx)
  );
  return {
    c() {
      g = svg_element("g");
      if (if_block0)
        if_block0.c();
      if_block0_anchor = empty$1();
      if (if_block1)
        if_block1.c();
      attr(
        g,
        "transform",
        /*transformOuterGroup*/
        ctx[5]()
      );
    },
    m(target, anchor2) {
      insert(target, g, anchor2);
      if (if_block0)
        if_block0.m(g, null);
      append$1(g, if_block0_anchor);
      if (if_block1)
        if_block1.m(g, null);
    },
    p(ctx2, [dirty]) {
      if (
        /*showAxisX*/
        ctx2[0]
      ) {
        if (if_block0) {
          if_block0.p(ctx2, dirty);
        } else {
          if_block0 = create_if_block_1$2(ctx2);
          if_block0.c();
          if_block0.m(g, if_block0_anchor);
        }
      } else if (if_block0) {
        if_block0.d(1);
        if_block0 = null;
      }
      if (
        /*showAxisY*/
        ctx2[1]
      ) {
        if (if_block1) {
          if_block1.p(ctx2, dirty);
        } else {
          if_block1 = create_if_block$5(ctx2);
          if_block1.c();
          if_block1.m(g, null);
        }
      } else if (if_block1) {
        if_block1.d(1);
        if_block1 = null;
      }
    },
    i: noop$2,
    o: noop$2,
    d(detaching) {
      if (detaching) {
        detach(g);
      }
      if (if_block0)
        if_block0.d();
      if (if_block1)
        if_block1.d();
    }
  };
}
function instance$c($$self, $$props, $$invalidate) {
  let { offsetX = 0 } = $$props;
  let { offsetY = 0 } = $$props;
  let { showAxisX = true } = $$props;
  let { showAxisY = true } = $$props;
  let { width } = $$props;
  let { height } = $$props;
  let { domainY } = $$props;
  let { domainX } = $$props;
  let { ticksY = 8 } = $$props;
  let { formatY = ".0%" } = $$props;
  let { ticksX = 8 } = $$props;
  let { formatX = ".3d" } = $$props;
  let axisXContainer;
  let axisYContainer;
  const renderAxisX = () => {
    if (!axisXContainer)
      return;
    const scaleX = linear().domain(domainX).range([0, width]);
    const axis2 = axisBottom(scaleX);
    axis2.ticks(ticksX, formatX);
    select(axisXContainer).call(axis2);
  };
  const renderAxisY = () => {
    if (!axisYContainer)
      return;
    const scaleY = linear().domain(domainY).range([height, 0]);
    const axis2 = axisLeft(scaleY);
    axis2.ticks(ticksY, formatY);
    select(axisYContainer).call(axis2);
  };
  afterUpdate(() => {
    renderAxisX();
    renderAxisY();
  });
  const transformOuterGroup = () => {
    if (offsetX === 0 && offsetY === 0)
      return "";
    return `translate(${offsetX ? offsetX : 0}, ${offsetY ? offsetY : 0})`;
  };
  function g_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      axisXContainer = $$value;
      $$invalidate(3, axisXContainer);
    });
  }
  function g_binding_1($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      axisYContainer = $$value;
      $$invalidate(4, axisYContainer);
    });
  }
  $$self.$$set = ($$props2) => {
    if ("offsetX" in $$props2)
      $$invalidate(6, offsetX = $$props2.offsetX);
    if ("offsetY" in $$props2)
      $$invalidate(7, offsetY = $$props2.offsetY);
    if ("showAxisX" in $$props2)
      $$invalidate(0, showAxisX = $$props2.showAxisX);
    if ("showAxisY" in $$props2)
      $$invalidate(1, showAxisY = $$props2.showAxisY);
    if ("width" in $$props2)
      $$invalidate(8, width = $$props2.width);
    if ("height" in $$props2)
      $$invalidate(2, height = $$props2.height);
    if ("domainY" in $$props2)
      $$invalidate(9, domainY = $$props2.domainY);
    if ("domainX" in $$props2)
      $$invalidate(10, domainX = $$props2.domainX);
    if ("ticksY" in $$props2)
      $$invalidate(11, ticksY = $$props2.ticksY);
    if ("formatY" in $$props2)
      $$invalidate(12, formatY = $$props2.formatY);
    if ("ticksX" in $$props2)
      $$invalidate(13, ticksX = $$props2.ticksX);
    if ("formatX" in $$props2)
      $$invalidate(14, formatX = $$props2.formatX);
  };
  return [
    showAxisX,
    showAxisY,
    height,
    axisXContainer,
    axisYContainer,
    transformOuterGroup,
    offsetX,
    offsetY,
    width,
    domainY,
    domainX,
    ticksY,
    formatY,
    ticksX,
    formatX,
    g_binding,
    g_binding_1
  ];
}
class SingleAxisLayer extends SvelteComponent {
  constructor(options) {
    super();
    init$1(this, options, instance$c, create_fragment$d, safe_not_equal, {
      offsetX: 6,
      offsetY: 7,
      showAxisX: 0,
      showAxisY: 1,
      width: 8,
      height: 2,
      domainY: 9,
      domainX: 10,
      ticksY: 11,
      formatY: 12,
      ticksX: 13,
      formatX: 14
    });
  }
}
function get_each_context$6(ctx, list2, i2) {
  const child_ctx = ctx.slice();
  child_ctx[16] = list2[i2];
  return child_ctx;
}
function create_each_block$6(ctx) {
  let path;
  let path_d_value;
  let g0;
  let text_1;
  let t_value = (
    /*chart*/
    ctx[16].label + ""
  );
  let t;
  let g0_transform_value;
  let g1;
  let line0;
  let line0_y__value;
  let line0_y__value_1;
  let line1;
  let line1_x__value_1;
  let line1_y__value;
  let line1_y__value_1;
  let line2;
  let line2_y__value;
  let line2_y__value_1;
  let path_levels = [
    {
      d: path_d_value = /*chart*/
      ctx[16].pathString
    },
    /*style*/
    ctx[3][
      /*chart*/
      ctx[16].ensemble
    ]
  ];
  let path_data = {};
  for (let i2 = 0; i2 < path_levels.length; i2 += 1) {
    path_data = assign(path_data, path_levels[i2]);
  }
  return {
    c() {
      path = svg_element("path");
      g0 = svg_element("g");
      text_1 = svg_element("text");
      t = text(t_value);
      g1 = svg_element("g");
      line0 = svg_element("line");
      line1 = svg_element("line");
      line2 = svg_element("line");
      set_svg_attributes(path, path_data);
      attr(text_1, "text-anchor", "end");
      attr(text_1, "dominant-baseline", "text-after-edge");
      attr(g0, "transform", g0_transform_value = `translate(${-8},${/*chart*/
      ctx[16].y1 - 5})`);
      attr(line0, "x", 0);
      attr(line0, "y1", line0_y__value = /*chart*/
      ctx[16].y0 + 20);
      attr(line0, "y2", line0_y__value_1 = /*chart*/
      ctx[16].y1);
      attr(line1, "x2", -30);
      attr(line1, "x1", line1_x__value_1 = /*width*/
      ctx[4] - /*axisMarginLeft*/
      ctx[1]);
      attr(line1, "y1", line1_y__value = /*chart*/
      ctx[16].y1);
      attr(line1, "y2", line1_y__value_1 = /*chart*/
      ctx[16].y1);
      attr(line2, "x2", -3);
      attr(line2, "x1", 3);
      attr(line2, "y1", line2_y__value = /*chart*/
      ctx[16].y0 + 20);
      attr(line2, "y2", line2_y__value_1 = /*chart*/
      ctx[16].y0 + 20);
      attr(g1, "stroke", "black");
      attr(g1, "stroke-dasharray", 1);
    },
    m(target, anchor2) {
      insert(target, path, anchor2);
      insert(target, g0, anchor2);
      append$1(g0, text_1);
      append$1(text_1, t);
      insert(target, g1, anchor2);
      append$1(g1, line0);
      append$1(g1, line1);
      append$1(g1, line2);
    },
    p(ctx2, dirty) {
      set_svg_attributes(path, path_data = get_spread_update(path_levels, [
        dirty & /*ridgelineCharts*/
        512 && path_d_value !== (path_d_value = /*chart*/
        ctx2[16].pathString) && { d: path_d_value },
        dirty & /*style, ridgelineCharts*/
        520 && /*style*/
        ctx2[3][
          /*chart*/
          ctx2[16].ensemble
        ]
      ]));
      if (dirty & /*ridgelineCharts*/
      512 && t_value !== (t_value = /*chart*/
      ctx2[16].label + ""))
        set_data(t, t_value);
      if (dirty & /*ridgelineCharts*/
      512 && g0_transform_value !== (g0_transform_value = `translate(${-8},${/*chart*/
      ctx2[16].y1 - 5})`)) {
        attr(g0, "transform", g0_transform_value);
      }
      if (dirty & /*ridgelineCharts*/
      512 && line0_y__value !== (line0_y__value = /*chart*/
      ctx2[16].y0 + 20)) {
        attr(line0, "y1", line0_y__value);
      }
      if (dirty & /*ridgelineCharts*/
      512 && line0_y__value_1 !== (line0_y__value_1 = /*chart*/
      ctx2[16].y1)) {
        attr(line0, "y2", line0_y__value_1);
      }
      if (dirty & /*width, axisMarginLeft*/
      18 && line1_x__value_1 !== (line1_x__value_1 = /*width*/
      ctx2[4] - /*axisMarginLeft*/
      ctx2[1])) {
        attr(line1, "x1", line1_x__value_1);
      }
      if (dirty & /*ridgelineCharts*/
      512 && line1_y__value !== (line1_y__value = /*chart*/
      ctx2[16].y1)) {
        attr(line1, "y1", line1_y__value);
      }
      if (dirty & /*ridgelineCharts*/
      512 && line1_y__value_1 !== (line1_y__value_1 = /*chart*/
      ctx2[16].y1)) {
        attr(line1, "y2", line1_y__value_1);
      }
      if (dirty & /*ridgelineCharts*/
      512 && line2_y__value !== (line2_y__value = /*chart*/
      ctx2[16].y0 + 20)) {
        attr(line2, "y1", line2_y__value);
      }
      if (dirty & /*ridgelineCharts*/
      512 && line2_y__value_1 !== (line2_y__value_1 = /*chart*/
      ctx2[16].y0 + 20)) {
        attr(line2, "y2", line2_y__value_1);
      }
    },
    d(detaching) {
      if (detaching) {
        detach(path);
        detach(g0);
        detach(g1);
      }
    }
  };
}
function create_fragment$c(ctx) {
  let div;
  let slider0;
  let updating_value;
  let t0;
  let slider1;
  let updating_value_1;
  let t1;
  let slider2;
  let updating_value_2;
  let portal_action;
  let t2;
  let svg;
  let g0;
  let g0_transform_value;
  let g1;
  let singleaxislayer;
  let g1_transform_value;
  let current;
  let mounted;
  let dispose;
  function slider0_value_binding(value) {
    ctx[13](value);
  }
  let slider0_props = {
    min: 0,
    max: 200,
    step: 1,
    valueDisplay: func$3
  };
  if (
    /*overlap*/
    ctx[8] !== void 0
  ) {
    slider0_props.value = /*overlap*/
    ctx[8];
  }
  slider0 = new Slider({ props: slider0_props });
  binding_callbacks.push(() => bind(slider0, "value", slider0_value_binding));
  function slider1_value_binding(value) {
    ctx[14](value);
  }
  let slider1_props = {
    min: 0.01,
    max: 2,
    step: 0.01,
    valueDisplay: func_1$2
  };
  if (
    /*bandwidth*/
    ctx[6] !== void 0
  ) {
    slider1_props.value = /*bandwidth*/
    ctx[6];
  }
  slider1 = new Slider({ props: slider1_props });
  binding_callbacks.push(() => bind(slider1, "value", slider1_value_binding));
  function slider2_value_binding(value) {
    ctx[15](value);
  }
  let slider2_props = {
    min: 2,
    max: 100,
    step: 1,
    valueDisplay: func_2$2
  };
  if (
    /*numPoints*/
    ctx[7] !== void 0
  ) {
    slider2_props.value = /*numPoints*/
    ctx[7];
  }
  slider2 = new Slider({ props: slider2_props });
  binding_callbacks.push(() => bind(slider2, "value", slider2_value_binding));
  let each_value = ensure_array_like(
    /*ridgelineCharts*/
    ctx[9]().areas.reverse()
  );
  let each_blocks = [];
  for (let i2 = 0; i2 < each_value.length; i2 += 1) {
    each_blocks[i2] = create_each_block$6(get_each_context$6(ctx, each_value, i2));
  }
  singleaxislayer = new SingleAxisLayer({
    props: {
      width: (
        /*width*/
        ctx[4] - /*axisMarginLeft*/
        ctx[1]
      ),
      height: (
        /*height*/
        ctx[5] - /*axisMarginBottom*/
        ctx[2]
      ),
      domainX: (
        /*ridgelineCharts*/
        ctx[9]().sharedDomainX
      ),
      domainY: (
        /*ridgelineCharts*/
        ctx[9]().sharedDomainY
      ),
      formatY: ".0",
      ticksY: 14,
      ticksX: 15,
      formatX: ".1f",
      showAxisY: false
    }
  });
  return {
    c() {
      div = element("div");
      create_component(slider0.$$.fragment);
      t0 = space();
      create_component(slider1.$$.fragment);
      t1 = space();
      create_component(slider2.$$.fragment);
      t2 = space();
      svg = svg_element("svg");
      g0 = svg_element("g");
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        each_blocks[i2].c();
      }
      g1 = svg_element("g");
      create_component(singleaxislayer.$$.fragment);
      attr(g0, "transform", g0_transform_value = `translate(${/*axisMarginLeft*/
      ctx[1]},0)`);
      attr(g1, "transform", g1_transform_value = `translate(${/*axisMarginLeft*/
      ctx[1]},0)`);
      attr(svg, "class", "absolute top-0 left-0 h-full w-full overflow-visible");
    },
    m(target, anchor2) {
      insert(target, div, anchor2);
      mount_component(slider0, div, null);
      append$1(div, t0);
      mount_component(slider1, div, null);
      append$1(div, t1);
      mount_component(slider2, div, null);
      insert(target, t2, anchor2);
      insert(target, svg, anchor2);
      append$1(svg, g0);
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        if (each_blocks[i2]) {
          each_blocks[i2].m(g0, null);
        }
      }
      append$1(svg, g1);
      mount_component(singleaxislayer, g1, null);
      current = true;
      if (!mounted) {
        dispose = action_destroyer(portal_action = portal.call(
          null,
          div,
          /*localControlsDivSelector*/
          ctx[0]
        ));
        mounted = true;
      }
    },
    p(ctx2, [dirty]) {
      const slider0_changes = {};
      if (!updating_value && dirty & /*overlap*/
      256) {
        updating_value = true;
        slider0_changes.value = /*overlap*/
        ctx2[8];
        add_flush_callback(() => updating_value = false);
      }
      slider0.$set(slider0_changes);
      const slider1_changes = {};
      if (!updating_value_1 && dirty & /*bandwidth*/
      64) {
        updating_value_1 = true;
        slider1_changes.value = /*bandwidth*/
        ctx2[6];
        add_flush_callback(() => updating_value_1 = false);
      }
      slider1.$set(slider1_changes);
      const slider2_changes = {};
      if (!updating_value_2 && dirty & /*numPoints*/
      128) {
        updating_value_2 = true;
        slider2_changes.value = /*numPoints*/
        ctx2[7];
        add_flush_callback(() => updating_value_2 = false);
      }
      slider2.$set(slider2_changes);
      if (portal_action && is_function(portal_action.update) && dirty & /*localControlsDivSelector*/
      1)
        portal_action.update.call(
          null,
          /*localControlsDivSelector*/
          ctx2[0]
        );
      if (dirty & /*ridgelineCharts, width, axisMarginLeft, style*/
      538) {
        each_value = ensure_array_like(
          /*ridgelineCharts*/
          ctx2[9]().areas.reverse()
        );
        let i2;
        for (i2 = 0; i2 < each_value.length; i2 += 1) {
          const child_ctx = get_each_context$6(ctx2, each_value, i2);
          if (each_blocks[i2]) {
            each_blocks[i2].p(child_ctx, dirty);
          } else {
            each_blocks[i2] = create_each_block$6(child_ctx);
            each_blocks[i2].c();
            each_blocks[i2].m(g0, null);
          }
        }
        for (; i2 < each_blocks.length; i2 += 1) {
          each_blocks[i2].d(1);
        }
        each_blocks.length = each_value.length;
      }
      if (!current || dirty & /*axisMarginLeft*/
      2 && g0_transform_value !== (g0_transform_value = `translate(${/*axisMarginLeft*/
      ctx2[1]},0)`)) {
        attr(g0, "transform", g0_transform_value);
      }
      const singleaxislayer_changes = {};
      if (dirty & /*width, axisMarginLeft*/
      18)
        singleaxislayer_changes.width = /*width*/
        ctx2[4] - /*axisMarginLeft*/
        ctx2[1];
      if (dirty & /*height, axisMarginBottom*/
      36)
        singleaxislayer_changes.height = /*height*/
        ctx2[5] - /*axisMarginBottom*/
        ctx2[2];
      if (dirty & /*ridgelineCharts*/
      512)
        singleaxislayer_changes.domainX = /*ridgelineCharts*/
        ctx2[9]().sharedDomainX;
      if (dirty & /*ridgelineCharts*/
      512)
        singleaxislayer_changes.domainY = /*ridgelineCharts*/
        ctx2[9]().sharedDomainY;
      singleaxislayer.$set(singleaxislayer_changes);
      if (!current || dirty & /*axisMarginLeft*/
      2 && g1_transform_value !== (g1_transform_value = `translate(${/*axisMarginLeft*/
      ctx2[1]},0)`)) {
        attr(g1, "transform", g1_transform_value);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(slider0.$$.fragment, local);
      transition_in(slider1.$$.fragment, local);
      transition_in(slider2.$$.fragment, local);
      transition_in(singleaxislayer.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(slider0.$$.fragment, local);
      transition_out(slider1.$$.fragment, local);
      transition_out(slider2.$$.fragment, local);
      transition_out(singleaxislayer.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div);
        detach(t2);
        detach(svg);
      }
      destroy_component(slider0);
      destroy_component(slider1);
      destroy_component(slider2);
      destroy_each(each_blocks, detaching);
      destroy_component(singleaxislayer);
      mounted = false;
      dispose();
    }
  };
}
const func$3 = (v) => `Overlap: ${v}px`;
const func_1$2 = (v) => `KDE Bandwidth: ${v}`;
const func_2$2 = (v) => `Number of points: ${v}`;
function instance$b($$self, $$props, $$invalidate) {
  let ridgelineCharts;
  let { localControlsDivSelector } = $$props;
  let { axisMarginLeft = 60 } = $$props;
  let { axisMarginBottom = 35 } = $$props;
  let { showAxisX = true } = $$props;
  let { showAxisY = true } = $$props;
  let { data } = $$props;
  let { style: style2 } = $$props;
  let { width } = $$props;
  let { height } = $$props;
  let bandwidth = 1.1;
  let numPoints = 50;
  let overlap = 20;
  function slider0_value_binding(value) {
    overlap = value;
    $$invalidate(8, overlap);
  }
  function slider1_value_binding(value) {
    bandwidth = value;
    $$invalidate(6, bandwidth);
  }
  function slider2_value_binding(value) {
    numPoints = value;
    $$invalidate(7, numPoints);
  }
  $$self.$$set = ($$props2) => {
    if ("localControlsDivSelector" in $$props2)
      $$invalidate(0, localControlsDivSelector = $$props2.localControlsDivSelector);
    if ("axisMarginLeft" in $$props2)
      $$invalidate(1, axisMarginLeft = $$props2.axisMarginLeft);
    if ("axisMarginBottom" in $$props2)
      $$invalidate(2, axisMarginBottom = $$props2.axisMarginBottom);
    if ("showAxisX" in $$props2)
      $$invalidate(10, showAxisX = $$props2.showAxisX);
    if ("showAxisY" in $$props2)
      $$invalidate(11, showAxisY = $$props2.showAxisY);
    if ("data" in $$props2)
      $$invalidate(12, data = $$props2.data);
    if ("style" in $$props2)
      $$invalidate(3, style2 = $$props2.style);
    if ("width" in $$props2)
      $$invalidate(4, width = $$props2.width);
    if ("height" in $$props2)
      $$invalidate(5, height = $$props2.height);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty & /*data, bandwidth, numPoints, height, overlap, axisMarginBottom, width, axisMarginLeft*/
    4598) {
      $$invalidate(9, ridgelineCharts = () => {
        const ensembles = new Set(data.map((d) => d.ensemble_id));
        const experiment_id = data[0].experiment;
        const experiment = getLoadedExperiments()[experiment_id];
        const sortedEnsembles = experiment.sortedEnsembles().filter((d) => ensembles.has(d.id));
        const kdeInfos = sortedEnsembles.map((ens) => {
          const valuesForEnsemble = data.filter((d) => d.ensemble_id === ens.id).map((d) => d.values);
          return {
            ...kde(valuesForEnsemble, bandwidth, kdeEpanechnikov(0.3), numPoints),
            ensemble: ens.id
          };
        });
        const sharedDomainY = [
          Math.min(...kdeInfos.map((d) => d.domainY[0])),
          Math.max(...kdeInfos.map((d) => d.domainY[1]))
        ];
        const sharedDomainX = [
          Math.min(...kdeInfos.map((d) => d.domainX[0])),
          Math.max(...kdeInfos.map((d) => d.domainX[1]))
        ];
        const allocatedHeight = height - overlap - axisMarginBottom;
        const heightPerArea = allocatedHeight / sortedEnsembles.length;
        const scaleName2y1 = band().domain(sortedEnsembles.map((d) => d.name)).range([allocatedHeight, 0]).padding(0).align(0);
        const scaleYNormalizedToShared = linear().domain([0, 1]).range(sharedDomainY);
        const scaleXNormalizedToShared = linear().domain([0, 1]).range(sharedDomainX);
        const scaleXSharedToViewport = linear().domain(sharedDomainX).range([0, width - axisMarginLeft]);
        const points = [];
        for (let iteration = 0; iteration < sortedEnsembles.length; ++iteration) {
          const ens = sortedEnsembles[iteration];
          const y1 = scaleName2y1(ens.name);
          const lineInfo = kdeInfos[iteration];
          const scaleY = linear().domain(sharedDomainY).range([y1 + heightPerArea, y1 - overlap]);
          const areaGenerator = area().x((d) => scaleXSharedToViewport(scaleXNormalizedToShared(d[0]))).y1((d) => scaleY(scaleYNormalizedToShared(d[1]))).y0(y1 + heightPerArea + overlap).curve(basis);
          const pathString = areaGenerator(lineInfo.kdeValues);
          points.push({
            pathString,
            ensemble: ens.id,
            label: `Iter ${ens.iteration}`,
            iteration: ens.iteration,
            x0: 0,
            y0: y1 + overlap,
            y1: y1 + overlap + heightPerArea,
            domainY: sharedDomainY,
            domainX: sharedDomainX,
            height: heightPerArea,
            rect: {
              x0: 0,
              y1,
              y0: y1 - heightPerArea,
              x1: width - axisMarginLeft
            }
          });
        }
        return {
          areas: points,
          sharedDomainX,
          sharedDomainY
        };
      });
    }
  };
  return [
    localControlsDivSelector,
    axisMarginLeft,
    axisMarginBottom,
    style2,
    width,
    height,
    bandwidth,
    numPoints,
    overlap,
    ridgelineCharts,
    showAxisX,
    showAxisY,
    data,
    slider0_value_binding,
    slider1_value_binding,
    slider2_value_binding
  ];
}
class ParameterKDERidgelines extends SvelteComponent {
  constructor(options) {
    super();
    init$1(this, options, instance$b, create_fragment$c, safe_not_equal, {
      localControlsDivSelector: 0,
      axisMarginLeft: 1,
      axisMarginBottom: 2,
      showAxisX: 10,
      showAxisY: 11,
      data: 12,
      style: 3,
      width: 4,
      height: 5
    });
  }
}
function create_fragment$b(ctx) {
  let g;
  let path;
  let path_d_value;
  let path_levels = [
    { class: "fill-none" },
    /*style*/
    ctx[0],
    { d: path_d_value = /*compute*/
    ctx[1]() }
  ];
  let path_data = {};
  for (let i2 = 0; i2 < path_levels.length; i2 += 1) {
    path_data = assign(path_data, path_levels[i2]);
  }
  return {
    c() {
      g = svg_element("g");
      path = svg_element("path");
      set_svg_attributes(path, path_data);
    },
    m(target, anchor2) {
      insert(target, g, anchor2);
      append$1(g, path);
    },
    p(ctx2, [dirty]) {
      set_svg_attributes(path, path_data = get_spread_update(path_levels, [
        { class: "fill-none" },
        dirty & /*style*/
        1 && /*style*/
        ctx2[0],
        dirty & /*compute*/
        2 && path_d_value !== (path_d_value = /*compute*/
        ctx2[1]()) && { d: path_d_value }
      ]));
    },
    i: noop$2,
    o: noop$2,
    d(detaching) {
      if (detaching) {
        detach(g);
      }
    }
  };
}
function instance$a($$self, $$props, $$invalidate) {
  let compute;
  let { width } = $$props;
  let { height } = $$props;
  let { domainY } = $$props;
  let { domainX } = $$props;
  let { sharedDomainX } = $$props;
  let { sharedDomainY } = $$props;
  let { points } = $$props;
  let { realization } = $$props;
  let { style: style2 } = $$props;
  $$self.$$set = ($$props2) => {
    if ("width" in $$props2)
      $$invalidate(2, width = $$props2.width);
    if ("height" in $$props2)
      $$invalidate(3, height = $$props2.height);
    if ("domainY" in $$props2)
      $$invalidate(4, domainY = $$props2.domainY);
    if ("domainX" in $$props2)
      $$invalidate(5, domainX = $$props2.domainX);
    if ("sharedDomainX" in $$props2)
      $$invalidate(6, sharedDomainX = $$props2.sharedDomainX);
    if ("sharedDomainY" in $$props2)
      $$invalidate(7, sharedDomainY = $$props2.sharedDomainY);
    if ("points" in $$props2)
      $$invalidate(8, points = $$props2.points);
    if ("realization" in $$props2)
      $$invalidate(9, realization = $$props2.realization);
    if ("style" in $$props2)
      $$invalidate(0, style2 = $$props2.style);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty & /*sharedDomainX, width, sharedDomainY, height, points*/
    460) {
      $$invalidate(1, compute = () => {
        const scaleXNormalizedToShared = linear().domain([0, 1]).range(sharedDomainX);
        const scaleXSharedToViewport = linear().domain(sharedDomainX).range([0, width]);
        const scaleYNormalizedToShared = linear().domain([0, 1]).range(sharedDomainY);
        const scaleYSharedToViewport = linear().domain(sharedDomainY).range([height, 0]);
        const scaleX = (x2) => scaleXSharedToViewport(scaleXNormalizedToShared(x2));
        const scaleY = (y2) => scaleYSharedToViewport(scaleYNormalizedToShared(y2));
        const lineGenerator = line$1().x((d) => scaleX(d[0])).y((d) => scaleY(d[1]));
        const lineDataString = lineGenerator(points);
        if (!lineDataString) {
          throw new Error("Failed to create line from data");
        }
        return lineDataString;
      });
    }
  };
  return [
    style2,
    compute,
    width,
    height,
    domainY,
    domainX,
    sharedDomainX,
    sharedDomainY,
    points,
    realization
  ];
}
class LineLayer extends SvelteComponent {
  constructor(options) {
    super();
    init$1(this, options, instance$a, create_fragment$b, safe_not_equal, {
      width: 2,
      height: 3,
      domainY: 4,
      domainX: 5,
      sharedDomainX: 6,
      sharedDomainY: 7,
      points: 8,
      realization: 9,
      style: 0
    });
  }
}
function get_each_context$5(ctx, list2, i2) {
  const child_ctx = ctx.slice();
  child_ctx[20] = list2[i2];
  child_ctx[22] = i2;
  return child_ctx;
}
function create_if_block$4(ctx) {
  let g;
  let singleaxislayer;
  let g_transform_value;
  let current;
  singleaxislayer = new SingleAxisLayer({
    props: {
      width: (
        /*width*/
        ctx[7] - /*axisMarginLeft*/
        ctx[1]
      ),
      height: (
        /*height*/
        ctx[8] - /*axisMarginBottom*/
        ctx[2]
      ),
      domainX: (
        /*sharedDomainX*/
        ctx[13]()
      ),
      domainY: (
        /*sharedDomainY*/
        ctx[14]()
      ),
      formatY: ".2f"
    }
  });
  return {
    c() {
      g = svg_element("g");
      create_component(singleaxislayer.$$.fragment);
      attr(g, "transform", g_transform_value = `translate(${/*axisMarginLeft*/
      ctx[1]},0)`);
    },
    m(target, anchor2) {
      insert(target, g, anchor2);
      mount_component(singleaxislayer, g, null);
      current = true;
    },
    p(ctx2, dirty) {
      const singleaxislayer_changes = {};
      if (dirty & /*width, axisMarginLeft*/
      130)
        singleaxislayer_changes.width = /*width*/
        ctx2[7] - /*axisMarginLeft*/
        ctx2[1];
      if (dirty & /*height, axisMarginBottom*/
      260)
        singleaxislayer_changes.height = /*height*/
        ctx2[8] - /*axisMarginBottom*/
        ctx2[2];
      if (dirty & /*sharedDomainX*/
      8192)
        singleaxislayer_changes.domainX = /*sharedDomainX*/
        ctx2[13]();
      if (dirty & /*sharedDomainY*/
      16384)
        singleaxislayer_changes.domainY = /*sharedDomainY*/
        ctx2[14]();
      singleaxislayer.$set(singleaxislayer_changes);
      if (!current || dirty & /*axisMarginLeft*/
      2 && g_transform_value !== (g_transform_value = `translate(${/*axisMarginLeft*/
      ctx2[1]},0)`)) {
        attr(g, "transform", g_transform_value);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(singleaxislayer.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(singleaxislayer.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(g);
      }
      destroy_component(singleaxislayer);
    }
  };
}
function create_each_block$5(ctx) {
  let linelayer;
  let current;
  linelayer = new LineLayer({
    props: {
      width: (
        /*width*/
        ctx[7] - /*axisMarginLeft*/
        ctx[1]
      ),
      height: (
        /*height*/
        ctx[8] - /*axisMarginBottom*/
        ctx[2]
      ),
      domainX: (
        /*chart*/
        ctx[20].domainX
      ),
      domainY: (
        /*chart*/
        ctx[20].domainY
      ),
      sharedDomainX: (
        /*sharedDomainX*/
        ctx[13]()
      ),
      sharedDomainY: (
        /*sharedDomainY*/
        ctx[14]()
      ),
      points: (
        /*chart*/
        ctx[20].points
      ),
      realization: (
        /*chart*/
        ctx[20].realization
      ),
      style: {
        .../*style*/
        ctx[6][
          /*chart*/
          ctx[20].ensemble
        ],
        "stroke-dashoffset": `${/*i*/
        ctx[22] * /*dashoffsetMultiplier*/
        ctx[11]}px`,
        "stroke-dasharray": (
          /*dasharray*/
          ctx[10].toString()
        ),
        "stroke-width": (
          /*thickness*/
          ctx[12]
        )
      }
    }
  });
  return {
    c() {
      create_component(linelayer.$$.fragment);
    },
    m(target, anchor2) {
      mount_component(linelayer, target, anchor2);
      current = true;
    },
    p(ctx2, dirty) {
      const linelayer_changes = {};
      if (dirty & /*width, axisMarginLeft*/
      130)
        linelayer_changes.width = /*width*/
        ctx2[7] - /*axisMarginLeft*/
        ctx2[1];
      if (dirty & /*height, axisMarginBottom*/
      260)
        linelayer_changes.height = /*height*/
        ctx2[8] - /*axisMarginBottom*/
        ctx2[2];
      if (dirty & /*data*/
      32)
        linelayer_changes.domainX = /*chart*/
        ctx2[20].domainX;
      if (dirty & /*data*/
      32)
        linelayer_changes.domainY = /*chart*/
        ctx2[20].domainY;
      if (dirty & /*sharedDomainX*/
      8192)
        linelayer_changes.sharedDomainX = /*sharedDomainX*/
        ctx2[13]();
      if (dirty & /*sharedDomainY*/
      16384)
        linelayer_changes.sharedDomainY = /*sharedDomainY*/
        ctx2[14]();
      if (dirty & /*data*/
      32)
        linelayer_changes.points = /*chart*/
        ctx2[20].points;
      if (dirty & /*data*/
      32)
        linelayer_changes.realization = /*chart*/
        ctx2[20].realization;
      if (dirty & /*style, data, dashoffsetMultiplier, dasharray, thickness*/
      7264)
        linelayer_changes.style = {
          .../*style*/
          ctx2[6][
            /*chart*/
            ctx2[20].ensemble
          ],
          "stroke-dashoffset": `${/*i*/
          ctx2[22] * /*dashoffsetMultiplier*/
          ctx2[11]}px`,
          "stroke-dasharray": (
            /*dasharray*/
            ctx2[10].toString()
          ),
          "stroke-width": (
            /*thickness*/
            ctx2[12]
          )
        };
      linelayer.$set(linelayer_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(linelayer.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(linelayer.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(linelayer, detaching);
    }
  };
}
function create_fragment$a(ctx) {
  let div;
  let slider0;
  let updating_value;
  let t0;
  let slider1;
  let updating_value_1;
  let t1;
  let slider2;
  let updating_value_2;
  let t2;
  let slider3;
  let updating_value_3;
  let portal_action;
  let t3;
  let svg;
  let g;
  let g_transform_value;
  let g_opacity_value;
  let current;
  let mounted;
  let dispose;
  function slider0_value_binding(value) {
    ctx[16](value);
  }
  let slider0_props = {
    min: 0.5,
    max: 4,
    step: 0.5,
    valueDisplay: func$2
  };
  if (
    /*thickness*/
    ctx[12] !== void 0
  ) {
    slider0_props.value = /*thickness*/
    ctx[12];
  }
  slider0 = new Slider({ props: slider0_props });
  binding_callbacks.push(() => bind(slider0, "value", slider0_value_binding));
  function slider1_value_binding(value) {
    ctx[17](value);
  }
  let slider1_props = {
    min: 1,
    max: 100,
    step: 1,
    valueDisplay: func_1$1
  };
  if (
    /*opacity*/
    ctx[9] !== void 0
  ) {
    slider1_props.value = /*opacity*/
    ctx[9];
  }
  slider1 = new Slider({ props: slider1_props });
  binding_callbacks.push(() => bind(slider1, "value", slider1_value_binding));
  function slider2_value_binding(value) {
    ctx[18](value);
  }
  let slider2_props = {
    min: 0,
    max: 10,
    step: 1,
    valueDisplay: func_2$1
  };
  if (
    /*dasharray*/
    ctx[10] !== void 0
  ) {
    slider2_props.value = /*dasharray*/
    ctx[10];
  }
  slider2 = new Slider({ props: slider2_props });
  binding_callbacks.push(() => bind(slider2, "value", slider2_value_binding));
  function slider3_value_binding(value) {
    ctx[19](value);
  }
  let slider3_props = {
    min: 1,
    max: 100,
    step: 1,
    valueDisplay: func_3$1
  };
  if (
    /*dashoffsetMultiplier*/
    ctx[11] !== void 0
  ) {
    slider3_props.value = /*dashoffsetMultiplier*/
    ctx[11];
  }
  slider3 = new Slider({ props: slider3_props });
  binding_callbacks.push(() => bind(slider3, "value", slider3_value_binding));
  let if_block = (
    /*showAxisX*/
    (ctx[3] || /*showAxisY*/
    ctx[4]) && create_if_block$4(ctx)
  );
  let each_value = ensure_array_like(
    /*data*/
    ctx[5]
  );
  let each_blocks = [];
  for (let i2 = 0; i2 < each_value.length; i2 += 1) {
    each_blocks[i2] = create_each_block$5(get_each_context$5(ctx, each_value, i2));
  }
  const out = (i2) => transition_out(each_blocks[i2], 1, 1, () => {
    each_blocks[i2] = null;
  });
  return {
    c() {
      div = element("div");
      create_component(slider0.$$.fragment);
      t0 = space();
      create_component(slider1.$$.fragment);
      t1 = space();
      create_component(slider2.$$.fragment);
      t2 = space();
      create_component(slider3.$$.fragment);
      t3 = space();
      svg = svg_element("svg");
      if (if_block)
        if_block.c();
      g = svg_element("g");
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        each_blocks[i2].c();
      }
      attr(g, "transform", g_transform_value = `translate(${/*axisMarginLeft*/
      ctx[1]},0)`);
      attr(g, "opacity", g_opacity_value = /*opacity*/
      ctx[9] / 100);
      attr(svg, "class", "absolute top-0 left-0 h-full w-full overflow-visible");
    },
    m(target, anchor2) {
      insert(target, div, anchor2);
      mount_component(slider0, div, null);
      append$1(div, t0);
      mount_component(slider1, div, null);
      append$1(div, t1);
      mount_component(slider2, div, null);
      append$1(div, t2);
      mount_component(slider3, div, null);
      insert(target, t3, anchor2);
      insert(target, svg, anchor2);
      if (if_block)
        if_block.m(svg, null);
      append$1(svg, g);
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        if (each_blocks[i2]) {
          each_blocks[i2].m(g, null);
        }
      }
      current = true;
      if (!mounted) {
        dispose = action_destroyer(portal_action = portal.call(
          null,
          div,
          /*localControlsDivSelector*/
          ctx[0]
        ));
        mounted = true;
      }
    },
    p(ctx2, [dirty]) {
      const slider0_changes = {};
      if (!updating_value && dirty & /*thickness*/
      4096) {
        updating_value = true;
        slider0_changes.value = /*thickness*/
        ctx2[12];
        add_flush_callback(() => updating_value = false);
      }
      slider0.$set(slider0_changes);
      const slider1_changes = {};
      if (!updating_value_1 && dirty & /*opacity*/
      512) {
        updating_value_1 = true;
        slider1_changes.value = /*opacity*/
        ctx2[9];
        add_flush_callback(() => updating_value_1 = false);
      }
      slider1.$set(slider1_changes);
      const slider2_changes = {};
      if (!updating_value_2 && dirty & /*dasharray*/
      1024) {
        updating_value_2 = true;
        slider2_changes.value = /*dasharray*/
        ctx2[10];
        add_flush_callback(() => updating_value_2 = false);
      }
      slider2.$set(slider2_changes);
      const slider3_changes = {};
      if (!updating_value_3 && dirty & /*dashoffsetMultiplier*/
      2048) {
        updating_value_3 = true;
        slider3_changes.value = /*dashoffsetMultiplier*/
        ctx2[11];
        add_flush_callback(() => updating_value_3 = false);
      }
      slider3.$set(slider3_changes);
      if (portal_action && is_function(portal_action.update) && dirty & /*localControlsDivSelector*/
      1)
        portal_action.update.call(
          null,
          /*localControlsDivSelector*/
          ctx2[0]
        );
      if (
        /*showAxisX*/
        ctx2[3] || /*showAxisY*/
        ctx2[4]
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
          if (dirty & /*showAxisX, showAxisY*/
          24) {
            transition_in(if_block, 1);
          }
        } else {
          if_block = create_if_block$4(ctx2);
          if_block.c();
          transition_in(if_block, 1);
          if_block.m(svg, g);
        }
      } else if (if_block) {
        group_outros();
        transition_out(if_block, 1, 1, () => {
          if_block = null;
        });
        check_outros();
      }
      if (dirty & /*width, axisMarginLeft, height, axisMarginBottom, data, sharedDomainX, sharedDomainY, style, dashoffsetMultiplier, dasharray, thickness*/
      32230) {
        each_value = ensure_array_like(
          /*data*/
          ctx2[5]
        );
        let i2;
        for (i2 = 0; i2 < each_value.length; i2 += 1) {
          const child_ctx = get_each_context$5(ctx2, each_value, i2);
          if (each_blocks[i2]) {
            each_blocks[i2].p(child_ctx, dirty);
            transition_in(each_blocks[i2], 1);
          } else {
            each_blocks[i2] = create_each_block$5(child_ctx);
            each_blocks[i2].c();
            transition_in(each_blocks[i2], 1);
            each_blocks[i2].m(g, null);
          }
        }
        group_outros();
        for (i2 = each_value.length; i2 < each_blocks.length; i2 += 1) {
          out(i2);
        }
        check_outros();
      }
      if (!current || dirty & /*axisMarginLeft*/
      2 && g_transform_value !== (g_transform_value = `translate(${/*axisMarginLeft*/
      ctx2[1]},0)`)) {
        attr(g, "transform", g_transform_value);
      }
      if (!current || dirty & /*opacity*/
      512 && g_opacity_value !== (g_opacity_value = /*opacity*/
      ctx2[9] / 100)) {
        attr(g, "opacity", g_opacity_value);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(slider0.$$.fragment, local);
      transition_in(slider1.$$.fragment, local);
      transition_in(slider2.$$.fragment, local);
      transition_in(slider3.$$.fragment, local);
      transition_in(if_block);
      for (let i2 = 0; i2 < each_value.length; i2 += 1) {
        transition_in(each_blocks[i2]);
      }
      current = true;
    },
    o(local) {
      transition_out(slider0.$$.fragment, local);
      transition_out(slider1.$$.fragment, local);
      transition_out(slider2.$$.fragment, local);
      transition_out(slider3.$$.fragment, local);
      transition_out(if_block);
      each_blocks = each_blocks.filter(Boolean);
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        transition_out(each_blocks[i2]);
      }
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div);
        detach(t3);
        detach(svg);
      }
      destroy_component(slider0);
      destroy_component(slider1);
      destroy_component(slider2);
      destroy_component(slider3);
      if (if_block)
        if_block.d();
      destroy_each(each_blocks, detaching);
      mounted = false;
      dispose();
    }
  };
}
const func$2 = (v) => `Line thickness: ${v}px`;
const func_1$1 = (v) => `Global line opacity: ${v}%`;
const func_2$1 = (v) => `Stipled by ${v}px`;
const func_3$1 = (v) => `Stipling spread: ${v}px`;
function instance$9($$self, $$props, $$invalidate) {
  let sharedDomainY;
  let sharedDomainX;
  let { localControlsDivSelector } = $$props;
  let { axisMarginLeft = 35 } = $$props;
  let { axisMarginBottom = 35 } = $$props;
  let { showAxisX = true } = $$props;
  let { showAxisY = true } = $$props;
  let { data } = $$props;
  let { style: style2 } = $$props;
  let { width } = $$props;
  let { height } = $$props;
  let { spec } = $$props;
  let opacity2 = 100;
  let dasharray = 0;
  let dashoffsetMultiplier = 1;
  let thickness = 0.5;
  function slider0_value_binding(value) {
    thickness = value;
    $$invalidate(12, thickness);
  }
  function slider1_value_binding(value) {
    opacity2 = value;
    $$invalidate(9, opacity2);
  }
  function slider2_value_binding(value) {
    dasharray = value;
    $$invalidate(10, dasharray);
  }
  function slider3_value_binding(value) {
    dashoffsetMultiplier = value;
    $$invalidate(11, dashoffsetMultiplier);
  }
  $$self.$$set = ($$props2) => {
    if ("localControlsDivSelector" in $$props2)
      $$invalidate(0, localControlsDivSelector = $$props2.localControlsDivSelector);
    if ("axisMarginLeft" in $$props2)
      $$invalidate(1, axisMarginLeft = $$props2.axisMarginLeft);
    if ("axisMarginBottom" in $$props2)
      $$invalidate(2, axisMarginBottom = $$props2.axisMarginBottom);
    if ("showAxisX" in $$props2)
      $$invalidate(3, showAxisX = $$props2.showAxisX);
    if ("showAxisY" in $$props2)
      $$invalidate(4, showAxisY = $$props2.showAxisY);
    if ("data" in $$props2)
      $$invalidate(5, data = $$props2.data);
    if ("style" in $$props2)
      $$invalidate(6, style2 = $$props2.style);
    if ("width" in $$props2)
      $$invalidate(7, width = $$props2.width);
    if ("height" in $$props2)
      $$invalidate(8, height = $$props2.height);
    if ("spec" in $$props2)
      $$invalidate(15, spec = $$props2.spec);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty & /*data*/
    32) {
      $$invalidate(14, sharedDomainY = () => {
        const yDomains = data.map((c) => c.domainY);
        return [
          Math.min(...yDomains.map((d) => d[0])),
          Math.max(...yDomains.map((d) => d[1]))
        ];
      });
    }
    if ($$self.$$.dirty & /*data*/
    32) {
      $$invalidate(13, sharedDomainX = () => {
        const xDomains = data.map((c) => c.domainX);
        return [
          Math.min(...xDomains.map((d) => d[0])),
          Math.max(...xDomains.map((d) => d[1]))
        ];
      });
    }
  };
  return [
    localControlsDivSelector,
    axisMarginLeft,
    axisMarginBottom,
    showAxisX,
    showAxisY,
    data,
    style2,
    width,
    height,
    opacity2,
    dasharray,
    dashoffsetMultiplier,
    thickness,
    sharedDomainX,
    sharedDomainY,
    spec,
    slider0_value_binding,
    slider1_value_binding,
    slider2_value_binding,
    slider3_value_binding
  ];
}
class SummaryLines extends SvelteComponent {
  constructor(options) {
    super();
    init$1(this, options, instance$9, create_fragment$a, safe_not_equal, {
      localControlsDivSelector: 0,
      axisMarginLeft: 1,
      axisMarginBottom: 2,
      showAxisX: 3,
      showAxisY: 4,
      data: 5,
      style: 6,
      width: 7,
      height: 8,
      spec: 15
    });
  }
}
const watchResize = (element2) => {
  const resizeObserver = new ResizeObserver((entries) => {
    element2.dispatchEvent(new CustomEvent("resized", { detail: { entries } }));
  });
  resizeObserver.observe(element2);
  return {
    destroy() {
      resizeObserver.disconnect();
    }
  };
};
function create_fragment$9(ctx) {
  let button;
  let current;
  let mounted;
  let dispose;
  const default_slot_template = (
    /*#slots*/
    ctx[3].default
  );
  const default_slot = create_slot(
    default_slot_template,
    ctx,
    /*$$scope*/
    ctx[2],
    null
  );
  return {
    c() {
      button = element("button");
      if (default_slot)
        default_slot.c();
      attr(button, "class", "flex content-start items-center min-h-10 w-full px-3 sidebar-button svelte-11cb3ce");
      attr(button, "role", "menuitem");
      attr(button, "tabindex", "0");
      toggle_class(
        button,
        "sidebar-button--active",
        /*selected*/
        ctx[0]
      );
    },
    m(target, anchor2) {
      insert(target, button, anchor2);
      if (default_slot) {
        default_slot.m(button, null);
      }
      current = true;
      if (!mounted) {
        dispose = listen(
          button,
          "click",
          /*click_handler*/
          ctx[4]
        );
        mounted = true;
      }
    },
    p(ctx2, [dirty]) {
      if (default_slot) {
        if (default_slot.p && (!current || dirty & /*$$scope*/
        4)) {
          update_slot_base(
            default_slot,
            default_slot_template,
            ctx2,
            /*$$scope*/
            ctx2[2],
            !current ? get_all_dirty_from_scope(
              /*$$scope*/
              ctx2[2]
            ) : get_slot_changes(
              default_slot_template,
              /*$$scope*/
              ctx2[2],
              dirty,
              null
            ),
            null
          );
        }
      }
      if (!current || dirty & /*selected*/
      1) {
        toggle_class(
          button,
          "sidebar-button--active",
          /*selected*/
          ctx2[0]
        );
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(default_slot, local);
      current = true;
    },
    o(local) {
      transition_out(default_slot, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(button);
      }
      if (default_slot)
        default_slot.d(detaching);
      mounted = false;
      dispose();
    }
  };
}
function instance$8($$self, $$props, $$invalidate) {
  let { $$slots: slots = {}, $$scope } = $$props;
  let { selected = false } = $$props;
  const dispatch2 = createEventDispatcher();
  const click_handler = (e) => dispatch2("click", e);
  $$self.$$set = ($$props2) => {
    if ("selected" in $$props2)
      $$invalidate(0, selected = $$props2.selected);
    if ("$$scope" in $$props2)
      $$invalidate(2, $$scope = $$props2.$$scope);
  };
  return [selected, dispatch2, $$scope, slots, click_handler];
}
class SidebarButton extends SvelteComponent {
  constructor(options) {
    super();
    init$1(this, options, instance$8, create_fragment$9, safe_not_equal, { selected: 0 });
  }
}
const get_controls_slot_changes = (dirty) => ({});
const get_controls_slot_context = (ctx) => ({});
function get_each_context$4(ctx, list2, i2) {
  const child_ctx = ctx.slice();
  child_ctx[8] = list2[i2];
  return child_ctx;
}
const get_open_button_slot_changes = (dirty) => ({});
const get_open_button_slot_context = (ctx) => ({});
function fallback_block_1(ctx) {
  let button;
  let mounted;
  let dispose;
  return {
    c() {
      button = element("button");
      button.textContent = "Controls";
      attr(button, "class", "h-10 w-20 bg-red");
    },
    m(target, anchor2) {
      insert(target, button, anchor2);
      if (!mounted) {
        dispose = listen(
          button,
          "click",
          /*toggleOpen*/
          ctx[3]
        );
        mounted = true;
      }
    },
    p: noop$2,
    d(detaching) {
      if (detaching) {
        detach(button);
      }
      mounted = false;
      dispose();
    }
  };
}
function create_default_slot(ctx) {
  let t_value = (
    /*exp*/
    ctx[8].name + ""
  );
  let t;
  return {
    c() {
      t = text(t_value);
    },
    m(target, anchor2) {
      insert(target, t, anchor2);
    },
    p: noop$2,
    d(detaching) {
      if (detaching) {
        detach(t);
      }
    }
  };
}
function create_each_block$4(ctx) {
  let li;
  let sidebarbutton;
  let t;
  let current;
  function click_handler() {
    return (
      /*click_handler*/
      ctx[5](
        /*exp*/
        ctx[8]
      )
    );
  }
  sidebarbutton = new SidebarButton({
    props: {
      selected: (
        /*experiment*/
        ctx[0] === /*exp*/
        ctx[8].id
      ),
      $$slots: { default: [create_default_slot] },
      $$scope: { ctx }
    }
  });
  sidebarbutton.$on("click", click_handler);
  return {
    c() {
      li = element("li");
      create_component(sidebarbutton.$$.fragment);
      t = space();
    },
    m(target, anchor2) {
      insert(target, li, anchor2);
      mount_component(sidebarbutton, li, null);
      append$1(li, t);
      current = true;
    },
    p(new_ctx, dirty) {
      ctx = new_ctx;
      const sidebarbutton_changes = {};
      if (dirty & /*experiment*/
      1)
        sidebarbutton_changes.selected = /*experiment*/
        ctx[0] === /*exp*/
        ctx[8].id;
      if (dirty & /*$$scope*/
      64) {
        sidebarbutton_changes.$$scope = { dirty, ctx };
      }
      sidebarbutton.$set(sidebarbutton_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(sidebarbutton.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(sidebarbutton.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(li);
      }
      destroy_component(sidebarbutton);
    }
  };
}
function fallback_block$1(ctx) {
  let t;
  return {
    c() {
      t = text('Add controls to the \\"controls\\" slot');
    },
    m(target, anchor2) {
      insert(target, t, anchor2);
    },
    d(detaching) {
      if (detaching) {
        detach(t);
      }
    }
  };
}
function create_fragment$8(ctx) {
  let span;
  let t0;
  let div2;
  let div0;
  let t1;
  let nav;
  let ul;
  let t2;
  let div1;
  let current;
  let mounted;
  let dispose;
  const open_button_slot_template = (
    /*#slots*/
    ctx[4]["open-button"]
  );
  const open_button_slot = create_slot(
    open_button_slot_template,
    ctx,
    /*$$scope*/
    ctx[6],
    get_open_button_slot_context
  );
  const open_button_slot_or_fallback = open_button_slot || fallback_block_1(ctx);
  let each_value = ensure_array_like(Object.values(
    /*experiments*/
    ctx[2]
  ));
  let each_blocks = [];
  for (let i2 = 0; i2 < each_value.length; i2 += 1) {
    each_blocks[i2] = create_each_block$4(get_each_context$4(ctx, each_value, i2));
  }
  const out = (i2) => transition_out(each_blocks[i2], 1, 1, () => {
    each_blocks[i2] = null;
  });
  const controls_slot_template = (
    /*#slots*/
    ctx[4].controls
  );
  const controls_slot = create_slot(
    controls_slot_template,
    ctx,
    /*$$scope*/
    ctx[6],
    get_controls_slot_context
  );
  const controls_slot_or_fallback = controls_slot || fallback_block$1();
  return {
    c() {
      span = element("span");
      if (open_button_slot_or_fallback)
        open_button_slot_or_fallback.c();
      t0 = space();
      div2 = element("div");
      div0 = element("div");
      t1 = space();
      nav = element("nav");
      ul = element("ul");
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        each_blocks[i2].c();
      }
      t2 = space();
      div1 = element("div");
      if (controls_slot_or_fallback)
        controls_slot_or_fallback.c();
      attr(div0, "class", "fixed top-0 left-0 h-dvh w-dvw sidebar-backdrop svelte-1mtj1d8");
      toggle_class(
        div0,
        "sidebar-backdrop--visible",
        /*open*/
        ctx[1]
      );
      attr(ul, "class", "p-2 min-w-60 sidebar__experiments svelte-1mtj1d8");
      attr(div1, "class", "flex flex-col gap-4 px-3 min-w-96");
      attr(nav, "class", "fixed flex top-0 left-0 h-dvh sidebar svelte-1mtj1d8");
      toggle_class(
        nav,
        "sidebar--open",
        /*open*/
        ctx[1]
      );
    },
    m(target, anchor2) {
      insert(target, span, anchor2);
      if (open_button_slot_or_fallback) {
        open_button_slot_or_fallback.m(span, null);
      }
      append$1(span, t0);
      append$1(span, div2);
      append$1(div2, div0);
      append$1(div2, t1);
      append$1(div2, nav);
      append$1(nav, ul);
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        if (each_blocks[i2]) {
          each_blocks[i2].m(ul, null);
        }
      }
      append$1(nav, t2);
      append$1(nav, div1);
      if (controls_slot_or_fallback) {
        controls_slot_or_fallback.m(div1, null);
      }
      current = true;
      if (!mounted) {
        dispose = [
          listen(
            div0,
            "click",
            /*toggleOpen*/
            ctx[3]
          ),
          action_destroyer(portal.call(null, div2, "#global-portal"))
        ];
        mounted = true;
      }
    },
    p(ctx2, [dirty]) {
      if (open_button_slot) {
        if (open_button_slot.p && (!current || dirty & /*$$scope*/
        64)) {
          update_slot_base(
            open_button_slot,
            open_button_slot_template,
            ctx2,
            /*$$scope*/
            ctx2[6],
            !current ? get_all_dirty_from_scope(
              /*$$scope*/
              ctx2[6]
            ) : get_slot_changes(
              open_button_slot_template,
              /*$$scope*/
              ctx2[6],
              dirty,
              get_open_button_slot_changes
            ),
            get_open_button_slot_context
          );
        }
      }
      if (!current || dirty & /*open*/
      2) {
        toggle_class(
          div0,
          "sidebar-backdrop--visible",
          /*open*/
          ctx2[1]
        );
      }
      if (dirty & /*experiment, Object, experiments*/
      5) {
        each_value = ensure_array_like(Object.values(
          /*experiments*/
          ctx2[2]
        ));
        let i2;
        for (i2 = 0; i2 < each_value.length; i2 += 1) {
          const child_ctx = get_each_context$4(ctx2, each_value, i2);
          if (each_blocks[i2]) {
            each_blocks[i2].p(child_ctx, dirty);
            transition_in(each_blocks[i2], 1);
          } else {
            each_blocks[i2] = create_each_block$4(child_ctx);
            each_blocks[i2].c();
            transition_in(each_blocks[i2], 1);
            each_blocks[i2].m(ul, null);
          }
        }
        group_outros();
        for (i2 = each_value.length; i2 < each_blocks.length; i2 += 1) {
          out(i2);
        }
        check_outros();
      }
      if (controls_slot) {
        if (controls_slot.p && (!current || dirty & /*$$scope*/
        64)) {
          update_slot_base(
            controls_slot,
            controls_slot_template,
            ctx2,
            /*$$scope*/
            ctx2[6],
            !current ? get_all_dirty_from_scope(
              /*$$scope*/
              ctx2[6]
            ) : get_slot_changes(
              controls_slot_template,
              /*$$scope*/
              ctx2[6],
              dirty,
              get_controls_slot_changes
            ),
            get_controls_slot_context
          );
        }
      }
      if (!current || dirty & /*open*/
      2) {
        toggle_class(
          nav,
          "sidebar--open",
          /*open*/
          ctx2[1]
        );
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(open_button_slot_or_fallback, local);
      for (let i2 = 0; i2 < each_value.length; i2 += 1) {
        transition_in(each_blocks[i2]);
      }
      transition_in(controls_slot_or_fallback, local);
      current = true;
    },
    o(local) {
      transition_out(open_button_slot_or_fallback, local);
      each_blocks = each_blocks.filter(Boolean);
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        transition_out(each_blocks[i2]);
      }
      transition_out(controls_slot_or_fallback, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(span);
      }
      if (open_button_slot_or_fallback)
        open_button_slot_or_fallback.d(detaching);
      destroy_each(each_blocks, detaching);
      if (controls_slot_or_fallback)
        controls_slot_or_fallback.d(detaching);
      mounted = false;
      run_all(dispose);
    }
  };
}
function instance$7($$self, $$props, $$invalidate) {
  let { $$slots: slots = {}, $$scope } = $$props;
  const experiments = getLoadedExperiments();
  let { experiment } = $$props;
  let { open = false } = $$props;
  const backdropCloseListener = (e) => {
    if (e.code === "Escape") {
      $$invalidate(1, open = false);
      window.removeEventListener("keypress", backdropCloseListener);
    }
  };
  const toggleOpen = () => {
    if (!open)
      window.addEventListener("keypress", backdropCloseListener);
    else
      window.removeEventListener("keypress", backdropCloseListener);
    $$invalidate(1, open = !open);
  };
  const click_handler = (exp) => $$invalidate(0, experiment = exp.id);
  $$self.$$set = ($$props2) => {
    if ("experiment" in $$props2)
      $$invalidate(0, experiment = $$props2.experiment);
    if ("open" in $$props2)
      $$invalidate(1, open = $$props2.open);
    if ("$$scope" in $$props2)
      $$invalidate(6, $$scope = $$props2.$$scope);
  };
  return [experiment, open, experiments, toggleOpen, slots, click_handler, $$scope];
}
class Sidebar extends SvelteComponent {
  constructor(options) {
    super();
    init$1(this, options, instance$7, create_fragment$8, safe_not_equal, { experiment: 0, open: 1 });
  }
}
var open_side_sheet = {
  name: "open_side_sheet",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21 4H3c-.55 0-1 .45-1 1v14c0 .55.45 1 1 1h18c.55 0 1-.45 1-1V5c0-.55-.45-1-1-1Zm-1 14H10V6h10v12Zm-4.84-7H11v2h4.16l-1.59 1.59L14.99 16 19 12.01 14.99 8l-1.41 1.41L15.16 11Z"
};
var reduce = {
  name: "reduce",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 4h2v2H3V4ZM7 6V4h2v2H7ZM3 8v2h2V8H3ZM3 13v9h9v-9H3Zm2 7v-5h5v5H5ZM17 22h-2v-2h2v2ZM19 20v2h2v-2h-2ZM21 16v2h-2v-2h2ZM21 10h-4.512l4.25-4.25-1.414-1.415L15 8.66V4h-2v8h8v-2Z"
};
var filter_alt_active = {
  name: "filter_alt_active",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M9.962 13s-3.73-4.8-5.75-7.39A.998.998 0 0 1 5.002 4h13.91c.83 0 1.3.95.79 1.61-2.02 2.59-5.74 7.39-5.74 7.39v6c0 .55-.45 1-1 1h-2c-.55 0-1-.45-1-1v-6Z"
};
var layers_off = {
  name: "layers_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m3.34 1.895-1.41 1.41 4.22 4.22-3.22 2.51 9 7 2.1-1.63 1.42 1.42-3.53 2.75-7.37-5.73-1.62 1.26 9 7 4.95-3.85 3.78 3.78 1.41-1.41L3.34 1.895Zm14.33 8.14-5.74-4.47-1.17.91-1.42-1.42 2.59-2.02 9 7-3.72 2.89-1.43-1.42 1.89-1.47Zm1.63 3.8 1.63 1.27-.87.68-1.43-1.43.67-.52Zm-13.11-3.8 5.74 4.47.67-.53-5.02-5.02-1.39 1.08Z"
};
var layers = {
  name: "layers",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m12 16.465 7.36-5.73L21 9.465l-9-7-9 7 1.63 1.27 7.37 5.73Zm-.01 2.54-7.37-5.73L3 14.535l9 7 9-7-1.63-1.27-7.38 5.74Zm5.75-9.54L12 4.995l-5.74 4.47 5.74 4.47 5.74-4.47Z"
};
var more_horizontal = {
  name: "more_horizontal",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M6 10c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2Zm12 0c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2Zm-8 2c0-1.1.9-2 2-2s2 .9 2 2-.9 2-2 2-2-.9-2-2Z"
};
var more_vertical = {
  name: "more_vertical",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2Zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2Zm-2 8c0-1.1.9-2 2-2s2 .9 2 2-.9 2-2 2-2-.9-2-2Z"
};
var fullscreen_exit = {
  name: "fullscreen_exit",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 8h3V5h2v5H5V8Zm3 8H5v-2h5v5H8v-3Zm6 3h2v-3h3v-2h-5v5Zm2-14v3h3v2h-5V5h2Z"
};
var vertical_split = {
  name: "vertical_split",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11 5H3v2h8V5ZM3 9h8v2H3V9Zm8 4H3v2h8v-2Zm0 4H3v2h8v-2Zm8-10v10h-4V7h4Zm-6-2h8v14h-8V5Z"
};
var view_agenda = {
  name: "view_agenda",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20.5 3h-17c-.55 0-1 .45-1 1v6c0 .55.45 1 1 1h17c.55 0 1-.45 1-1V4c0-.55-.45-1-1-1Zm-1 6V5h-15v4h15Zm0 10v-4h-15v4h15Zm-16-6h17c.55 0 1 .45 1 1v6c0 .55-.45 1-1 1h-17c-.55 0-1-.45-1-1v-6c0-.55.45-1 1-1Z"
};
var view_array = {
  name: "view_array",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3.5 5.5h3v13h-3v-13Zm13 0h-9v13h9v-13Zm1 0h3v13h-3v-13Zm-3 11v-9h-5v9h5Z"
};
var view_carousel = {
  name: "view_carousel",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17 19.5H7v-15h10v15ZM6 6.5H2v11h4v-11Zm3 0h6v11H9v-11Zm13 0h-4v11h4v-11Z"
};
var view_column = {
  name: "view_column",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3.5 18.5v-13h17v13h-17Zm10-2v-9h-3v9h3Zm-8-9h3v9h-3v-9Zm10 9h3v-9h-3v9Z"
};
var view_day = {
  name: "view_day",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2.5 4h19v2h-19V4Zm18 4h-17c-.55 0-1 .45-1 1v6c0 .55.45 1 1 1h17c.55 0 1-.45 1-1V9c0-.55-.45-1-1-1Zm-1 6v-4h-15v4h15Zm-17 4h19v2h-19v-2Z"
};
var view_list = {
  name: "view_list",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3.5 5v14h17V5h-17Zm4 2v2h-2V7h2Zm-2 4v2h2v-2h-2Zm0 4h2v2h-2v-2Zm4 2h9v-2h-9v2Zm9-4h-9v-2h9v2Zm-9-4h9V7h-9v2Z"
};
var view_module = {
  name: "view_module",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3.5 5.5v13h17v-13h-17Zm10 2V11h-3V7.5h3Zm-5 0h-3V11h3V7.5Zm-3 9V13h3v3.5h-3Zm5-3.5v3.5h3V13h-3Zm8 3.5h-3V13h3v3.5Zm-3-9V11h3V7.5h-3Z"
};
var view_quilt = {
  name: "view_quilt",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3.5 5.5v13h17v-13h-17Zm2 11v-9h3v9h-3Zm5-3.5v3.5h3V13h-3Zm8 3.5h-3V13h3v3.5Zm-8-9V11h8V7.5h-8Z"
};
var view_stream = {
  name: "view_stream",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3.5 6v12h17V6h-17Zm15 10h-13v-3h13v3Zm-13-8v3h13V8h-13Z"
};
var view_week = {
  name: "view_week",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 4h18c.55 0 1 .45 1 1v14c0 .55-.45 1-1 1H3c-.55 0-1-.45-1-1V5c0-.55.45-1 1-1Zm1 14h4V6H4v12Zm10 0h-4V6h4v12Zm2 0h4V6h-4v12Z"
};
var grid_off = {
  name: "grid_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M.564 1.98 1.974.57l21.46 21.45-1.41 1.41-2.01-2.009H4.565c-1.1 0-2-.9-2-1.999V3.978l-2-1.999Zm8 1.449v.89l2 1.998V3.43h4v3.998h-2.89l2 2h.89v.889l2 1.999V9.426h4v3.998h-2.89l2 2h.89v.89l2 1.998V3.43c0-1.1-.9-2-2-2H5.674l2 2h.89Zm8 0h4v3.998h-4V3.429Zm-6 8.546 1.45 1.45h-1.45v-1.45Zm-4.55-4.548-1.45-1.45v1.45h1.45Zm2.55 11.995h-4v-3.998h4v3.998Zm-4-5.998h4V9.976l-.55-.55h-3.45v3.998Zm10 5.998h-4v-3.998h3.45l.55.55v3.448Zm2-1.45v1.45h1.45l-1.45-1.45Z"
};
var grid_on = {
  name: "grid_on",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 2h16c1.1 0 2 .9 2 2v16c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2Zm0 18h4v-4H4v4Zm4-6H4v-4h4v4ZM4 8h4V4H4v4Zm10 12h-4v-4h4v4Zm-4-6h4v-4h-4v4Zm4-6h-4V4h4v4Zm2 12h4v-4h-4v4Zm4-6h-4v-4h4v4Zm-4-6h4V4h-4v4Z"
};
var dashboard = {
  name: "dashboard",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 3h8v10H3V3Zm18 0h-8v6h8V3ZM9 11V5H5v6h4Zm10-4V5h-4v2h4Zm0 6v6h-4v-6h4ZM9 19v-2H5v2h4Zm12-8h-8v10h8V11ZM3 15h8v6H3v-6Z"
};
var maximize = {
  name: "maximize",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 11h18v2H3v-2Z"
};
var minimize = {
  name: "minimize",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M6 11h12v2H6v-2Z"
};
var reorder = {
  name: "reorder",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 7V5h18v2H3Zm0 4h18V9H3v2Zm18 4H3v-2h18v2Zm0 4H3v-2h18v2Z"
};
var toc = {
  name: "toc",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17 9H3V7h14v2Zm0 4H3v-2h14v2ZM3 17h14v-2H3v2Zm18 0h-2v-2h2v2ZM19 7v2h2V7h-2Zm2 6h-2v-2h2v2Z"
};
var zoom_in = {
  name: "zoom_in",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15.756 14.255h-.79l-.28-.27a6.471 6.471 0 0 0 1.57-4.23 6.5 6.5 0 1 0-6.5 6.5c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99 1.49-1.49-4.99-5Zm-6 0c-2.49 0-4.5-2.01-4.5-4.5s2.01-4.5 4.5-4.5 4.5 2.01 4.5 4.5-2.01 4.5-4.5 4.5Zm-.5-5v-2h1v2h2v1h-2v2h-1v-2h-2v-1h2Z"
};
var zoom_out = {
  name: "zoom_out",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15.756 14.255h-.79l-.28-.27a6.471 6.471 0 0 0 1.57-4.23 6.5 6.5 0 1 0-6.5 6.5c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99 1.49-1.49-4.99-5Zm-6 0c-2.49 0-4.5-2.01-4.5-4.5s2.01-4.5 4.5-4.5 4.5 2.01 4.5 4.5-2.01 4.5-4.5 4.5Zm2.5-5h-5v1h5v-1Z"
};
var all_out = {
  name: "all_out",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 8V4h4L4 8Zm16 0-4-4h4v4Zm0 12v-4l-4 4h4ZM8 20H4v-4l4 4Zm11-8c0-3.87-3.13-7-7-7s-7 3.13-7 7 3.13 7 7 7 7-3.13 7-7ZM7 12c0 2.76 2.24 5 5 5s5-2.24 5-5-2.24-5-5-5-5 2.24-5 5Z"
};
var pan_tool = {
  name: "pan_tool",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11.82 23.995h6.55c2.21 0 4-1.79 4-4V6.145a2.5 2.5 0 0 0-3-2.45v-.28a2.5 2.5 0 0 0-2.5-2.5c-.33 0-.65.06-.94.18a2.48 2.48 0 0 0-2.06-1.09c-1.32 0-2.4 1.03-2.49 2.33a2.5 2.5 0 0 0-3.01 2.45v9.55l-2.41-1.28c-.73-.39-1.64-.28-2.26.27l-2.07 1.83 7.3 7.61c.75.78 1.81 1.23 2.89 1.23Zm-1.45-2.62-5.86-6.1.51-.45 5.35 2.83V4.785c0-.27.22-.5.5-.5s.5.22.5.5v7.21h2v-9.49c0-.28.22-.5.5-.5s.5.22.5.5v9.49h2v-8.58c0-.28.22-.5.5-.5s.5.22.5.5v8.58h2v-5.85c0-.28.22-.5.5-.5s.5.22.5.5v13.85c0 1.1-.9 2-2 2h-6.56c-.54 0-1.06-.23-1.44-.62Z"
};
var list = {
  name: "list",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 7h2v2H3V7Zm2 4H3v2h2v-2Zm16 0H7v2h14v-2ZM3 15h2v2H3v-2Zm18 0H7v2h14v-2Zm0-8H7v2h14V7Z"
};
var sort_by_alpha = {
  name: "sort_by_alpha",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m10.495 4.66 2.36-2.36 2.36 2.36h-4.72Zm4.69 14.71-2.33 2.33-2.33-2.33h4.66Zm-13.31-1.64 4.5-11.46h1.64l4.49 11.46h-1.84l-.92-2.45h-5.11l-.92 2.45h-1.84Zm3.37-4.09 1.94-5.18 1.94 5.18h-3.88Zm16.88 2.5h-6.12l5.93-8.6V6.28h-8.3v1.6h5.88l-5.92 8.56v1.29h8.53v-1.59Z"
};
var tune = {
  name: "tune",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17 9h-2V3h2v2h4v2h-4v2ZM3 7V5h10v2H3Zm0 12v-2h6v2H3Zm10 2v-2h8v-2h-8v-2h-2v6h2ZM7 11V9h2v6H7v-2H3v-2h4Zm14 2v-2H11v2h10Z"
};
var focus_center = {
  name: "focus_center",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 5h4V3H5c-1.1 0-2 .9-2 2v4h2V5Zm0 10H3v4c0 1.1.9 2 2 2h4v-2H5v-4ZM15 3h4c1.1 0 2 .9 2 2v4h-2V5h-4V3Zm4 16h-4v2h4c1.1 0 2-.9 2-2v-4h-2v4ZM9 12c0-1.66 1.34-3 3-3s3 1.34 3 3-1.34 3-3 3-3-1.34-3-3Z"
};
var compare = {
  name: "compare",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M10 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h5v2h2V1h-2v2Zm0 15H5l5-6v6Zm4-15h5c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2h-5v-9l5 6V5h-5V3Z"
};
var details = {
  name: "details",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 20 3 4h18l-9 16Zm5.63-14H6.38L12 16l5.63-10Z"
};
var touch = {
  name: "touch",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m18.264 12.94-3.24-1.62c1.29-1 2.12-2.56 2.12-4.32 0-3.03-2.47-5.5-5.5-5.5a5.51 5.51 0 0 0-5.5 5.5c0 2.13 1.22 3.98 3 4.89v3.26l-1.84-.39-.1-.02c-.1-.02-.2-.03-.32-.03-.53 0-1.03.21-1.41.59l-1.4 1.42 5.09 5.09c.43.44 1.03.69 1.65.69h6.3c.98 0 1.81-.7 1.97-1.67l.8-4.71c.22-1.3-.43-2.58-1.62-3.18Zm-.35 2.85-.8 4.71h-6.3c-.09 0-.17-.04-.24-.1l-3.68-3.68 4.25.89V7c0-.28.22-.5.5-.5s.5.22.5.5v6h1.76l3.46 1.73c.4.2.62.63.55 1.06ZM11.644 3.5c-1.93 0-3.5 1.57-3.5 3.5 0 .95.38 1.81 1 2.44V7a2.5 2.5 0 0 1 5 0v2.44c.62-.63 1-1.49 1-2.44 0-1.93-1.57-3.5-3.5-3.5Z"
};
var change_history = {
  name: "change_history",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 20 12 4l10 16H2Zm16.39-2L12 7.77 5.61 18h12.78Z"
};
var track_changes = {
  name: "track_changes",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m19.07 4.93-1.41 1.41A8.014 8.014 0 0 1 20 12c0 4.42-3.58 8-8 8s-8-3.58-8-8c0-4.08 3.05-7.44 7-7.93v2.02C8.16 6.57 6 9.03 6 12c0 3.31 2.69 6 6 6s6-2.69 6-6c0-1.66-.67-3.16-1.76-4.24l-1.41 1.41C15.55 9.9 16 10.9 16 12c0 2.21-1.79 4-4 4s-4-1.79-4-4c0-1.86 1.28-3.41 3-3.86v2.14c-.6.35-1 .98-1 1.72 0 1.1.9 2 2 2s2-.9 2-2c0-.74-.4-1.38-1-1.72V2h-1C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10c0-2.76-1.12-5.26-2.93-7.07Z"
};
var work = {
  name: "work",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16 6.5h4c1.11 0 2 .89 2 2v11c0 1.11-.89 2-2 2H4c-1.11 0-2-.89-2-2l.01-11c0-1.11.88-2 1.99-2h4v-2c0-1.11.89-2 2-2h4c1.11 0 2 .89 2 2v2Zm-6 0h4v-2h-4v2Z"
};
var work_off = {
  name: "work_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2.83 1.42 1.42 2.83l2.75 2.75h-.74c-1.11 0-1.99.89-1.99 2l-.01 11c0 1.11.89 2 2 2h15.74l2 2 1.41-1.41L2.83 1.42Zm6.6 2.16h4v2h-3.6l2 2h7.6v7.6l2 2v-9.6c0-1.11-.89-2-2-2h-4v-2c0-1.11-.89-2-2-2h-4c-.99 0-1.8.7-1.96 1.64l1.96 1.96v-1.6Zm-6 4v11h13.74l-11-11H3.43Z"
};
var work_outline = {
  name: "work_outline",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M22 8.5c0-1.11-.89-2-2-2h-4v-2c0-1.11-.89-2-2-2h-4c-1.11 0-2 .89-2 2v2H4c-1.11 0-1.99.89-1.99 2L2 19.5c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2v-11Zm-8-2v-2h-4v2h4Zm-10 2v11h16v-11H4Z"
};
var sort = {
  name: "sort",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 6v2h18V6H3Zm0 12h6v-2H3v2Zm12-5H3v-2h12v2Z"
};
var filter_list = {
  name: "filter_list",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 6v2h18V6H3Zm7 12h4v-2h-4v2Zm8-5H6v-2h12v2Z"
};
var filter_alt = {
  name: "filter_alt",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 6h10l-5.01 6.3L7 6Zm-2.75-.39C6.27 8.2 10 13 10 13v6c0 .55.45 1 1 1h2c.55 0 1-.45 1-1v-6s3.72-4.8 5.74-7.39A.998.998 0 0 0 18.95 4H5.04c-.83 0-1.3.95-.79 1.61Z"
};
var boundaries = {
  name: "boundaries",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m7.05 14.121-1.415 1.415L2.1 12l3.535-3.535L7.05 9.879 4.928 12l2.121 2.121Zm1.414 4.243 1.414-1.414 2.121 2.121 2.121-2.121 1.415 1.414-3.536 3.536-3.535-3.536ZM19.07 12l-2.121 2.121 1.414 1.415L21.9 12l-3.536-3.535-1.414 1.414L19.07 12ZM9.878 7.05 8.464 5.636l3.535-3.535 3.536 3.535L14.12 7.05 12 4.93 9.877 7.05Z"
};
var invert = {
  name: "invert",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11.553 2.106a1 1 0 0 1 .894 0l8 4A1 1 0 0 1 21 7v10a1 1 0 0 1-.553.894l-8 4a1 1 0 0 1-.894 0l-8-4A1 1 0 0 1 3 17V7a1 1 0 0 1 .553-.894l8-4ZM5 8.618 12 12v7.882l-7-3.5V8.618Zm7 1.264L6.236 7 12 4.118v5.764Z"
};
var inspect_rotation = {
  name: "inspect_rotation",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 5a8 8 0 1 0 5.646 2.333l1.412-1.417A9.972 9.972 0 0 1 22 13c0 5.523-4.477 10-10 10S2 18.523 2 13 6.477 3 12 3v2Z"
};
var inspect_3d = {
  name: "inspect_3d",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7.261 6.035c-.482.012-.758.148-.919.308-.3.3-.505 1.064-.018 2.5.457 1.346 1.436 2.969 2.865 4.553L10.585 12v4.243H6.342l1.431-1.43-.033-.037C6.16 13.034 5 11.165 4.43 9.486c-.542-1.6-.658-3.4.498-4.557.62-.62 1.445-.873 2.284-.893.831-.021 1.736.18 2.646.53.692.267 1.413.629 2.141 1.076.859-.527 1.708-.937 2.514-1.21 1.6-.543 3.4-.66 4.557.497.62.62.873 1.445.894 2.283.02.832-.182 1.737-.531 2.647A13.19 13.19 0 0 1 18.358 12c.447.728.808 1.45 1.075 2.141.35.91.551 1.815.53 2.647-.02.839-.272 1.663-.893 2.283-.666.667-1.564.907-2.467.895-.898-.013-1.88-.272-2.865-.697l.791-1.836c.834.359 1.549.525 2.101.533.548.007.855-.138 1.026-.309.16-.16.296-.437.308-.919.012-.489-.108-1.125-.398-1.879-.13-.338-.291-.691-.481-1.054a20.865 20.865 0 0 1-1.55 1.73c-1.833 1.833-3.855 3.198-5.677 3.898-.91.35-1.815.552-2.647.532-.838-.021-1.662-.273-2.283-.894-.666-.666-.907-1.565-.894-2.467.012-.898.271-1.88.696-2.865l1.837.791c-.36.834-.526 1.549-.534 2.101-.007.548.138.855.31 1.026.16.16.436.296.918.308.49.012 1.125-.108 1.88-.398 1.504-.58 3.297-1.764 4.98-3.446a18.438 18.438 0 0 0 1.827-2.12 18.448 18.448 0 0 0-1.828-2.122A18.436 18.436 0 0 0 12 8.05c-.464.34-.932.72-1.397 1.139L12 10.586H7.757V6.343l1.43 1.43.037-.032c.32-.29.644-.566.97-.827-.363-.19-.716-.35-1.054-.48-.754-.29-1.39-.411-1.879-.399Zm6.543.88c.587.468 1.168.987 1.73 1.55a20.852 20.852 0 0 1 1.55 1.73c.191-.363.352-.716.482-1.054.29-.754.41-1.39.398-1.879-.012-.482-.148-.758-.308-.919-.3-.3-1.063-.505-2.5-.018a9.57 9.57 0 0 0-1.352.59Z"
};
var fault = {
  name: "fault",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m18.752 10.1-2.107 3.65-2.126 3.715 4.729-2.172-1.562-.926 1.054-1.825 1.053-1.825-1.041-.617ZM4.602 13.9l2.107-3.65 2.126-3.715-4.73 2.172 1.563.926-1.054 1.825-1.054 1.825 1.042.617ZM11.618 5H3V3h10.353a1 1 0 0 1 .865 1.501l-7.53 13a1 1 0 0 1-.864.499H3v-2h2.247l6.371-11ZM11.735 19h8.619v2H10a1 1 0 0 1-.866-1.501l7.53-13A1 1 0 0 1 17.53 6h2.823v2h-2.247l-6.37 11Z"
};
var grid_layer = {
  name: "grid_layer",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20.412 13.451 12.001 20l-8.423-6.549L1.715 12 12 4l10.285 8-1.874 1.451Zm-6.14-4.791L12 6.89 9.583 8.774l2.421 1.775 2.268-1.89Zm1.845 1.437L18.561 12l-2.22 1.729-2.438-1.787 2.214-1.845Zm-5.922 1.96-2.486-1.823L5.44 12l2.493 1.941 2.261-1.884Zm-.416 3.322 2.315-1.93 2.372 1.74L12 17.109l-2.222-1.73Z"
};
var grid_layers = {
  name: "grid_layers",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m12 16.465 7.36-5.73L21 9.465l-9-7-9 7 1.63 1.27 7.37 5.73Zm-.01 2.54-7.37-5.73L3 14.535l9 7 9-7-1.63-1.27-7.38 5.74Zm1.997-12.463L12 4.995 9.885 6.642l2.118 1.553 1.984-1.653ZM15.602 7.8l2.138 1.665-1.943 1.513-2.132-1.564L15.602 7.8ZM10.42 9.515 8.245 7.919 6.26 9.465l2.181 1.699 1.98-1.65Zm-.364 2.906 2.026-1.688 2.075 1.522L12 13.935l-1.944-1.514Z"
};
var hill_shading = {
  name: "hill_shading",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14 3 9.78 8.63l1.25 1.67L14 6.33 19 13h-8.46L6.53 7.63 1 15h22L14 3ZM5 13l1.52-2.03L8.04 13H5Zm-2 6.5L2 17h2l1 2.5H3ZM8 21l-2-4h2l2 4H8Zm2-4 1 2.5h2L12 17h-2Zm4 0h2l2.5 5h-2L14 17Zm4 0 2 4h2l-2-4h-2Z"
};
var well = {
  name: "well",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M9.3 3h5.4v.9h1.8v1.8h-.9v1.717L17.784 19.2H19.2V21H4.8v-1.8h1.418L8.4 7.417V5.7h-.9V3.9h1.8V3Zm4.5 2.7v1.8l.005.023-1.804 1.11-1.804-1.11.004-.023V5.7h3.6Zm-3.778 2.772-.286 1.555 1.406-.865-1.12-.69Zm1.979 1.218-2.484 1.529-.298 1.62L12 14.386l2.782-1.545-.298-1.621L12 9.689Zm2.265.337-.286-1.555-1.12.69 1.406.865ZM11.074 14.9l-2.027-1.126-.462 2.509 2.49-1.383Zm.927.515 3.626 2.015.326 1.77H8.049l.325-1.77 3.627-2.015Zm3.415.868L12.927 14.9l2.028-1.126.461 2.509Z"
};
var surface_layer = {
  name: "surface_layer",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m12 20 8.412-6.549L22.286 12 12.001 4 1.715 12l1.863 1.451L12 20Zm0-13.109 5.212 4.06c-2.646 1.81-4.676 1.762-6.144 1.15-1.284-.535-2.215-1.532-2.781-2.317L12 6.89ZM18.56 12l-.522-.407c-2.996 2.148-5.467 2.218-7.355 1.431-1.49-.621-2.548-1.749-3.186-2.625L5.441 12l2.412 1.879c.84-.213 1.928-.348 3.028-.112 1.064.23 2.113.802 2.934 1.928L18.562 12Zm-5.534 4.31c-.678-.942-1.52-1.385-2.356-1.565a5.243 5.243 0 0 0-1.77-.051l3.1 2.415 1.026-.8Zm.224-6.292c-.002.054-.036.154-.193.258a1.196 1.196 0 0 1-.681.168 1.197 1.197 0 0 1-.67-.208c-.151-.113-.18-.214-.178-.27.002-.054.036-.154.194-.257.155-.103.395-.177.68-.169.286.008.521.097.67.209.151.112.18.214.178.269Zm-.904 1.426a2.19 2.19 0 0 0 1.26-.332c.345-.227.63-.594.643-1.065.014-.47-.248-.854-.58-1.1a2.191 2.191 0 0 0-1.237-.407 2.19 2.19 0 0 0-1.26.333c-.345.226-.63.593-.643 1.064-.014.47.248.854.58 1.1a2.19 2.19 0 0 0 1.237.407Z"
};
var miniplayer = {
  name: "miniplayer",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21 2H3c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h7v2H8v2h8v-2h-2v-2h7c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2Zm0 2v12H3V4h18Zm-8 5a1 1 0 0 0-1 1v4a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-4a1 1 0 0 0-1-1h-6Z"
};
var miniplayer_fullscreen = {
  name: "miniplayer_fullscreen",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 2h18c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2h-7v2h2v2H8v-2h2v-2H3c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2Zm0 2v12h18V4H3Zm16 9h1v2h-2v-1h1v-1Zm-7 0h1v1h1v1h-2v-2Zm8-4v2h-1v-1h-1V9h2Zm-8 2V9h2v1h-1v1h-1Z"
};
var fullscreen = {
  name: "fullscreen",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 10H5V5h5v2H7v3Zm-2 4h2v3h3v2H5v-5Zm12 3h-3v2h5v-5h-2v3ZM14 7V5h5v5h-2V7h-3Z",
  sizes: {
    small: {
      name: "fullscreen_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M5 7H3V3h4v2H5v2Zm-2 4h2v2h2v2H3v-4Zm10 2h-2v2h4v-4h-2v2Zm-2-8V3h4v4h-2V5h-2Z"
    }
  }
};
var expand_screen = {
  name: "expand_screen",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M13 5a1 1 0 0 1 1-1h5a1 1 0 0 1 1 1v5a1 1 0 1 1-2 0V7.414l-3.793 3.793a1 1 0 0 1-1.414-1.414L16.586 6H14a1 1 0 0 1-1-1ZM11.207 14.207a1 1 0 0 0-1.414-1.414L6 16.586V14a1 1 0 1 0-2 0v5a1 1 0 0 0 1 1h5a1 1 0 1 0 0-2H7.414l3.793-3.793Z"
};
var collapse_screen = {
  name: "collapse_screen",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20.207 4.293a1 1 0 0 1 0 1.414L16.414 9.5H19a1 1 0 1 1 0 2h-5a1 1 0 0 1-1-1v-5a1 1 0 1 1 2 0v2.586l3.793-3.793a1 1 0 0 1 1.414 0ZM4.5 14a1 1 0 0 1 1-1h5a1 1 0 0 1 1 1v5a1 1 0 1 1-2 0v-2.586l-3.793 3.793a1 1 0 0 1-1.414-1.414L8.086 15H5.5a1 1 0 0 1-1-1Z"
};
var filter_alt_off = {
  name: "filter_alt_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m3.414 2 17.352 17.352-1.414 1.414L14 15.414V19c0 .583-.424 1-1 1h-1.907C10.518 20 10 19.583 10 19v-6S6.768 8.865 4.41 5.825L2 3.415 3.414 2ZM19.74 5.61c-1.219 1.563-3.057 3.932-4.323 5.564l-1.41-1.41L17 6h-6.757l-2-2H18.95c.83 0 1.3.95.79 1.61Z"
};
var in_progress = {
  name: "in_progress",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 19.5v-15a7.5 7.5 0 1 0 0 15ZM12 2c5.523 0 10 4.477 10 10s-4.477 10-10 10S2 17.523 2 12 6.477 2 12 2Z"
};
var sheet_bottom_position = {
  name: "sheet_bottom_position",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 4h18c.55 0 1 .45 1 1v14c0 .55-.45 1-1 1H3c-.55 0-1-.45-1-1V5c0-.55.45-1 1-1Zm1 10h16V6H4v8Z"
};
var sheet_topposition = {
  name: "sheet_topposition",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 20h18c.55 0 1-.45 1-1V5c0-.55-.45-1-1-1H3c-.55 0-1 .45-1 1v14c0 .55.45 1 1 1Zm1-10h16v8H4v-8Z"
};
var sheet_leftposition = {
  name: "sheet_leftposition",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 20h18c.55 0 1-.45 1-1V5c0-.55-.45-1-1-1H3c-.55 0-1 .45-1 1v14c0 .55.45 1 1 1Zm7-14h10v12H10V6Z"
};
var sheet_rightposition = {
  name: "sheet_rightposition",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 4h18c.55 0 1 .45 1 1v14c0 .55-.45 1-1 1H3c-.55 0-1-.45-1-1V5c0-.55.45-1 1-1Zm1 14h10V6H4v12Z"
};
var enlarge = {
  name: "enlarge",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M9 3H3v18h18v-6h-2v4H5V5h4V3ZM10 12h2v2h-2v-2ZM6.375 14v-2h2.25v2h-2.25ZM10 15.375h2v2.25h-2v-2.25ZM17.512 5H13V3h8v8h-2V6.34l-4.324 4.325L13.26 9.25 17.512 5Z"
};
var signature = {
  name: "signature",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m17.37 2.29 2.34 2.34c.39.39.39 1.02 0 1.41l-1.83 1.83-3.75-3.75 1.83-1.83c.19-.19.44-.29.7-.29.26 0 .51.09.71.29ZM2 16.25V20h3.75L16.81 8.94l-3.75-3.75L2 16.25ZM4.92 18H4v-.92l9.06-9.06.92.92L4.92 18ZM10 18h9v2H8l2-2Z"
};
var select_all = {
  name: "select_all",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 5H3c0-1.1.9-2 2-2v2Zm0 8H3v-2h2v2Zm2 8h2v-2H7v2ZM5 9H3V7h2v2Zm8-6h-2v2h2V3Zm6 2V3c1.1 0 2 .9 2 2h-2ZM5 21v-2H3c0 1.1.9 2 2 2Zm0-4H3v-2h2v2ZM9 3H7v2h2V3Zm4 18h-2v-2h2v2Zm6-8h2v-2h-2v2Zm2 6c0 1.1-.9 2-2 2v-2h2ZM19 9h2V7h-2v2Zm2 8h-2v-2h2v2Zm-6 4h2v-2h-2v2Zm2-16h-2V3h2v2ZM7 17h10V7H7v10Zm8-8H9v6h6V9Z"
};
var unarchive = {
  name: "unarchive",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m19.15 3.55 1.39 1.68c.29.34.46.79.46 1.27V19c0 1.1-.9 2-2 2H5c-1.1 0-2-.9-2-2V6.5c0-.48.17-.93.46-1.27l1.38-1.68C5.12 3.21 5.53 3 6 3h12c.47 0 .88.21 1.15.55ZM17.76 5H6.24l-.82 1h13.17l-.83-1ZM5 19V8h14v11H5Zm5.55-5H8l4-4 4 4h-2.55v3h-2.9v-3Z"
};
var send = {
  name: "send",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m1.5 10 .01-7 20.99 9-20.99 9-.01-7 15-2-15-2Zm2.01-3.97 7.51 3.22-7.52-1 .01-2.22Zm7.5 8.72L3.5 17.97v-2.22l7.51-1Z"
};
var move_to_inbox = {
  name: "move_to_inbox",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4.99 3H19c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H4.99C3.88 21 3 20.1 3 19V5c0-1.1.88-2 1.99-2Zm8.46 6H16l-4 4-4-4h2.55V6h2.9v3ZM5 19v-3h3.56c.69 1.19 1.97 2 3.45 2 1.48 0 2.75-.81 3.45-2H19v3H5Zm9.01-5H19V5H4.99L5 14h5.01c0 1.1.9 2 2 2s2-.9 2-2Z"
};
var priority_low = {
  name: "priority_low",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M8.5 17.5C4.92 17.5 2 14.58 2 11s2.92-6.5 6.5-6.5H12v2H8.5C6.02 6.5 4 8.52 4 11s2.02 4.5 4.5 4.5H9V13l4 3.5L9 20v-2.5h-.5ZM22 4.5h-8v2h8v-2Zm0 5.5h-8v2h8v-2Zm-8 5.5h8v2h-8v-2Z"
};
var priority_high = {
  name: "priority_high",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M8.5 6.5C4.92 6.5 2 9.42 2 13s2.92 6.5 6.5 6.5H12v-2H8.5C6.02 17.5 4 15.48 4 13s2.02-4.5 4.5-4.5H9V11l4-3.5L9 4v2.5h-.5Zm13.5 0h-8v2h8v-2Zm0 5.5h-8v2h8v-2Zm-8 5.5h8v2h-8v-2Z"
};
var inbox = {
  name: "inbox",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 3H5c-1.1 0-2 .9-2 2v14a2 2 0 0 0 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2Zm0 16H5v-3h3.56c.69 1.19 1.97 2 3.45 2 1.48 0 2.75-.81 3.45-2H19v3Zm-4.99-5H19V5H5v9h5.01c0 1.1.9 2 2 2s2-.9 2-2Z"
};
var paste = {
  name: "paste",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 3h-4.18C14.4 1.84 13.3 1 12 1c-1.3 0-2.4.84-2.82 2H5c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2Zm-7 0c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1ZM5 5v16h14V5h-2v3H7V5H5Z"
};
var file_copy = {
  name: "file_copy",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16.5 1h-12c-1.1 0-2 .9-2 2v14h2V3h12V1Zm-1 4h-7c-1.1 0-1.99.9-1.99 2L6.5 21c0 1.1.89 2 1.99 2H19.5c1.1 0 2-.9 2-2V11l-6-6Zm-7 2v14h11v-9h-5V7h-6Z"
};
var delete_multiple = {
  name: "delete_multiple",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M6 4h4l1 1h3v2H2V5h3l1-1ZM5 20c-1.1 0-2-.9-2-2V8h10v10c0 1.1-.9 2-2 2H5ZM22 8h-7v2h7V8Zm-3 8h-4v2h4v-2Zm-4-4h6v2h-6v-2ZM5 10h6v8H5v-8Z"
};
var cut = {
  name: "cut",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M9.64 7.64c.23-.5.36-1.05.36-1.64 0-2.21-1.79-4-4-4S2 3.79 2 6s1.79 4 4 4c.59 0 1.14-.13 1.64-.36L10 12l-2.36 2.36C7.14 14.13 6.59 14 6 14c-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4c0-.59-.13-1.14-.36-1.64L12 14l7 7h3v-1L9.64 7.64ZM6 8a2 2 0 1 1-.001-3.999A2 2 0 0 1 6 8ZM4 18a2 2 0 1 0 3.999.001A2 2 0 0 0 4 18Zm8-5.5c-.28 0-.5-.22-.5-.5s.22-.5.5-.5.5.22.5.5-.22.5-.5.5ZM13 9l6-6h3v1l-7 7-2-2Z"
};
var edit = {
  name: "edit",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m18.368 3.29 2.34 2.34c.39.39.39 1.02 0 1.41l-1.83 1.83-3.75-3.75 1.83-1.83c.19-.19.44-.29.7-.29.26 0 .51.09.71.29ZM2.998 17.25V21h3.75l11.06-11.06-3.75-3.75-11.06 11.06ZM5.918 19h-.92v-.92l9.06-9.06.92.92L5.918 19Z"
};
var copy = {
  name: "copy",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16.5 1h-12c-1.1 0-2 .9-2 2v14h2V3h12V1Zm3 4h-11c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2Zm-11 16h11V7h-11v14Z"
};
var block = {
  name: "block",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2ZM4 12c0-4.42 3.58-8 8-8 1.85 0 3.55.63 4.9 1.69L5.69 16.9A7.902 7.902 0 0 1 4 12Zm3.1 6.31A7.902 7.902 0 0 0 12 20c4.42 0 8-3.58 8-8 0-1.85-.63-3.55-1.69-4.9L7.1 18.31Z"
};
var clear = {
  name: "clear",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41Z"
};
var archive = {
  name: "archive",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m19.15 3.55 1.39 1.68c.29.34.46.79.46 1.27V19c0 1.1-.9 2-2 2H5c-1.1 0-2-.9-2-2V6.5c0-.48.17-.93.46-1.27l1.38-1.68C5.12 3.21 5.53 3 6 3h12c.47 0 .88.21 1.15.55ZM17.76 5H6.24l-.8.97h13.13L17.76 5ZM5 19V8h14v11H5Zm5.55-9h2.9v3H16l-4 4-4-4h2.55v-3Z"
};
var add_circle_outlined = {
  name: "add_circle_outlined",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2Zm-1 5v4H7v2h4v4h2v-4h4v-2h-4V7h-2Zm-7 5c0 4.41 3.59 8 8 8s8-3.59 8-8-3.59-8-8-8-8 3.59-8 8Z"
};
var add_circle_filled = {
  name: "add_circle_filled",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10-4.48 10-10 10S2 17.52 2 12Zm11 1h4v-2h-4V7h-2v4H7v2h4v4h2v-4Z"
};
var add_box = {
  name: "add_box",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2Zm0 16H5V5h14v14Zm-6-2h-2v-4H7v-2h4V7h2v4h4v2h-4v4Z"
};
var add = {
  name: "add",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2Z"
};
var save = {
  name: "save",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 3h12l4 4v12c0 1.1-.9 2-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Zm14 16V7.83L16.17 5H5v14h14Zm-7-7c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3Zm3-6H6v4h9V6Z"
};
var report_off = {
  name: "report_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2.7 1.29 1.29 2.7l3.64 3.64-1.64 1.64v7.46l5.27 5.27h7.46l1.64-1.64 3.64 3.64 1.41-1.41L2.7 1.29Zm6.69 3.42h5.8l4.1 4.1v5.8l-.22.22 1.42 1.41.8-.8V7.98l-5.27-5.27H8.56l-.8.8 1.41 1.42.22-.22Zm2.9 10a1 1 0 1 0 0 2 1 1 0 0 0 0-2Zm1-8v2.33l-2-2v-.33h2Zm-3.9 12h5.8l1.05-1.05-9.9-9.9-1.05 1.05v5.8l4.1 4.1Z"
};
var remove_outlined = {
  name: "remove_outlined",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2ZM4 12c0 4.41 3.59 8 8 8s8-3.59 8-8-3.59-8-8-8-8 3.59-8 8Zm3-1v2h10v-2H7Z"
};
var remove = {
  name: "remove",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10-4.48 10-10 10S2 17.52 2 12Zm5 1h10v-2H7v2Z"
};
var report = {
  name: "report",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M8.27 3h7.46L21 8.27v7.46L15.73 21H8.27L3 15.73V8.27L8.27 3Zm6.63 16 4.1-4.1V9.1L14.9 5H9.1L5 9.1v5.8L9.1 19h5.8ZM12 15a1 1 0 1 0 0 2 1 1 0 0 0 0-2Zm1-8h-2v7h2V7Z"
};
var reply_all = {
  name: "reply_all",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 4.5v3l-4 4 4 4v3l-7-7 7-7Zm6 0v4c7 1 10 6 11 11-2.5-3.5-6-5.1-11-5.1v4.1l-7-7 7-7Z"
};
var reply = {
  name: "reply",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M10 8.5v-4l-7 7 7 7v-4.1c5 0 8.5 1.6 11 5.1-1-5-4-10-11-11Z"
};
var undo = {
  name: "undo",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12.266 8.5c-2.65 0-5.05.99-6.9 2.6l-3.6-3.6v9h9l-3.62-3.62c1.39-1.16 3.16-1.88 5.12-1.88 3.54 0 6.55 2.31 7.6 5.5l2.37-.78c-1.39-4.19-5.32-7.22-9.97-7.22Z"
};
var redo = {
  name: "redo",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18.63 11.1c-1.85-1.61-4.25-2.6-6.9-2.6-4.65 0-8.58 3.03-9.96 7.22l2.36.78a8.002 8.002 0 0 1 7.6-5.5c1.95 0 3.73.72 5.12 1.88l-3.62 3.62h9v-9l-3.6 3.6Z"
};
var refresh = {
  name: "refresh",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17.646 6.35A7.958 7.958 0 0 0 11.996 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08a5.99 5.99 0 0 1-5.65 4c-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L12.996 11h7V4l-2.35 2.35Z"
};
var loop = {
  name: "loop",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 1v3c4.42 0 8 3.58 8 8 0 1.57-.46 3.03-1.24 4.26L17.3 14.8c.45-.83.7-1.79.7-2.8 0-3.31-2.69-6-6-6v3L8 5l4-4ZM6 12c0 3.31 2.69 6 6 6v-3l4 4-4 4v-3c-4.42 0-8-3.58-8-8 0-1.57.46-3.03 1.24-4.26L6.7 9.2c-.45.83-.7 1.79-.7 2.8Z"
};
var autorenew = {
  name: "autorenew",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 9V6c-3.31 0-6 2.69-6 6 0 1.01.25 1.97.7 2.8l-1.46 1.46A7.93 7.93 0 0 1 4 12c0-4.42 3.58-8 8-8V1l4 4-4 4Zm5.3.2 1.46-1.46A7.93 7.93 0 0 1 20 12c0 4.42-3.58 8-8 8v3l-4-4 4-4v3c3.31 0 6-2.69 6-6 0-1.01-.26-1.96-.7-2.8Z"
};
var search_in_page = {
  name: "search_in_page",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6ZM6 4h7l5 5v8.58l-1.84-1.84a4.992 4.992 0 0 0-.64-6.28A4.96 4.96 0 0 0 12 8a5 5 0 0 0-3.53 1.46 4.98 4.98 0 0 0 0 7.05 4.982 4.982 0 0 0 6.28.63L17.6 20H6V4Zm6 11.98c.8 0 1.55-.32 2.11-.88.57-.56.88-1.31.88-2.11 0-.8-.32-1.55-.88-2.11-.56-.57-1.31-.88-2.11-.88-.8 0-1.55.31-2.11.88-.57.56-.88 1.31-.88 2.11 0 .8.32 1.55.88 2.11.56.57 1.31.88 2.11.88Z"
};
var search_find_replace = {
  name: "search_find_replace",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M13.796 6.715c-.91-.9-2.16-1.46-3.54-1.46a5 5 0 0 0-4.9 4h-2.02c.49-3.39 3.39-6 6.92-6 1.93 0 3.68.78 4.95 2.05l2.05-2.05v6h-6l2.54-2.54Zm3.38 4.54a6.89 6.89 0 0 1-1.28 3.14l4.85 4.86-1.49 1.49-4.86-4.85a6.984 6.984 0 0 1-4.14 1.36c-1.93 0-3.68-.78-4.95-2.05l-2.05 2.05v-6h6l-2.54 2.54c.91.9 2.16 1.46 3.54 1.46a5 5 0 0 0 4.9-4h2.02Z"
};
var history = {
  name: "history",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4.5 12a9 9 0 1 1 9 9c-2.49 0-4.73-1.01-6.36-2.64l1.42-1.42A6.944 6.944 0 0 0 13.5 19c3.87 0 7-3.13 7-7s-3.13-7-7-7-7 3.13-7 7h3l-4.04 4.03-.07-.14L1.5 12h3Zm8 1V8H14v4.15l3.52 2.09-.77 1.28L12.5 13Z"
};
var update = {
  name: "update",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21 3v7h-7l2.95-2.95A7.018 7.018 0 0 0 12 5c-3.86 0-7 3.14-7 7s3.14 7 7 7 7-3.14 7-7h2a9 9 0 1 1-9-9c2.49 0 4.74 1.01 6.36 2.64L21 3ZM11 13V8h1.5v4.15l3.52 2.09-.77 1.28L11 13Z"
};
var restore = {
  name: "restore",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4.5 12a9 9 0 1 1 9 9c-2.49 0-4.73-1.01-6.36-2.64l1.42-1.42A6.944 6.944 0 0 0 13.5 19c3.87 0 7-3.13 7-7s-3.13-7-7-7-7 3.13-7 7h3l-4 3.99-4-3.99h3Zm8 1V8H14v4.15l3.52 2.09-.77 1.28L12.5 13Z"
};
var restore_page = {
  name: "restore_page",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6Zm4 18H6V4h7.17L18 8.83V20ZM7.28 9.4l1.17 1.17c.8-1.24 2.18-2.07 3.77-2.07 2.48 0 4.5 2.02 4.5 4.5s-2.02 4.5-4.5 4.5a4.51 4.51 0 0 1-4.12-2.7h1.55a3.14 3.14 0 0 0 2.58 1.35 3.15 3.15 0 1 0 0-6.3c-1.21 0-2.27.7-2.79 1.71L10.88 13h-3.6V9.4Z"
};
var setting_backup_restore = {
  name: "setting_backup_restore",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4.5 12a9 9 0 1 1 3.52 7.14l1.42-1.44A6.995 6.995 0 0 0 20.5 12c0-3.87-3.13-7-7-7s-7 3.13-7 7h3l-4 4-4-4h3Zm9-2c1.1 0 2 .9 2 2s-.9 2-2 2-2-.9-2-2 .9-2 2-2Z"
};
var searched_history = {
  name: "searched_history",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17.01 14.26h-.8l-.27-.27a6.452 6.452 0 0 0 1.57-4.23 6.5 6.5 0 0 0-6.5-6.5c-3.59 0-6.5 3-6.5 6.5H2l3.84 4 4.16-4H6.51a4.5 4.5 0 0 1 9 0 4.507 4.507 0 0 1-6.32 4.12l-1.48 1.48a6.474 6.474 0 0 0 7.52-.67l.27.27v.79l5.01 4.99L22 19.26l-4.99-5Z"
};
var favorite_filled = {
  name: "favorite_filled",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m12 21.175-1.45-1.32C5.4 15.185 2 12.105 2 8.325c0-3.08 2.42-5.5 5.5-5.5 1.74 0 3.41.81 4.5 2.09 1.09-1.28 2.76-2.09 4.5-2.09 3.08 0 5.5 2.42 5.5 5.5 0 3.78-3.4 6.86-8.55 11.54L12 21.175Z"
};
var favorite_outlined = {
  name: "favorite_outlined",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 4.915c1.09-1.28 2.76-2.09 4.5-2.09 3.08 0 5.5 2.42 5.5 5.5 0 3.777-3.394 6.855-8.537 11.519l-.013.011-1.45 1.32-1.45-1.31-.04-.036C5.384 15.17 2 12.095 2 8.325c0-3.08 2.42-5.5 5.5-5.5 1.74 0 3.41.81 4.5 2.09Zm0 13.56.1-.1c4.76-4.31 7.9-7.16 7.9-10.05 0-2-1.5-3.5-3.5-3.5-1.54 0-3.04.99-3.56 2.36h-1.87c-.53-1.37-2.03-2.36-3.57-2.36-2 0-3.5 1.5-3.5 3.5 0 2.89 3.14 5.74 7.9 10.05l.1.1Z"
};
var star_filled = {
  name: "star_filled",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m12 16.067 4.947 3.6-1.894-5.814L20 10.334h-6.067l-1.933-6-1.933 6H4l4.947 3.52-1.894 5.814 4.947-3.6Z"
};
var star_half = {
  name: "star_half",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m22 9.24-7.19-.62L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21 12 17.27 18.18 21l-1.63-7.03L22 9.24ZM12 15.4V6.1l1.71 4.04 4.38.38-3.32 2.88 1 4.28L12 15.4Z"
};
var star_circle = {
  name: "star_circle",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2Zm7.48 7.16-5.01-.43-2-4.71c3.21.19 5.91 2.27 7.01 5.14Zm-5.07 6.26L12 13.98l-2.39 1.44.63-2.72-2.11-1.83 2.78-.24L12 8.06l1.09 2.56 2.78.24-2.11 1.83.64 2.73Zm-2.86-11.4-2 4.72-5.02.43c1.1-2.88 3.8-4.97 7.02-5.15ZM4 12c0-.64.08-1.26.23-1.86l3.79 3.28-1.11 4.75A7.982 7.982 0 0 1 4 12Zm3.84 6.82L12 16.31l4.16 2.5A7.924 7.924 0 0 1 11.99 20c-1.52 0-2.94-.44-4.15-1.18Zm9.25-.65-1.11-4.75 3.79-3.28c.14.59.23 1.22.23 1.86 0 2.48-1.14 4.7-2.91 6.17Z"
};
var star_outlined = {
  name: "star_outlined",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m22 9.24-7.19-.62L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21 12 17.27 18.18 21l-1.63-7.03L22 9.24ZM12 15.4l-3.76 2.27 1-4.28-3.32-2.88 4.38-.38L12 6.1l1.71 4.04 4.38.38-3.32 2.88 1 4.28L12 15.4Z"
};
var bookmarks = {
  name: "bookmarks",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2ZM9 4h2v5l-1-.75L9 9V4ZM6 20h12V4h-5v9l-3-2.25L7 13V4H6v16Z"
};
var bookmark_filled = {
  name: "bookmark_filled",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17 3H7c-1.1 0-2 .9-2 2v16l7-3 7 3V5c0-1.1-.9-2-2-2Z"
};
var bookmark_outlined = {
  name: "bookmark_outlined",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 3h10c1.1 0 2 .9 2 2v16l-7-3-7 3V5c0-1.1.9-2 2-2Zm5 12.82L17 18V5H7v13l5-2.18Z"
};
var bookmark_collection = {
  name: "bookmark_collection",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 1H8.99C7.89 1 7 1.9 7 3h10c1.1 0 2 .9 2 2v13l2 1V3c0-1.1-.9-2-2-2Zm-4 6v12.97l-4.21-1.81-.79-.34-.79.34L5 19.97V7h10ZM5 5h10c1.1 0 2 .9 2 2v16l-7-3-7 3V7c0-1.1.9-2 2-2Z"
};
var delete_to_trash = {
  name: "delete_to_trash",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14.5 3h-5l-1 1H5v2h14V4h-3.5l-1-1ZM16 9v10H8V9h8ZM6 7h12v12c0 1.1-.9 2-2 2H8c-1.1 0-2-.9-2-2V7Z"
};
var delete_forever = {
  name: "delete_forever",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m14.5 3 1 1H19v2H5V4h3.5l1-1h5ZM12 12.59l2.12-2.12 1.41 1.41L13.41 14l2.12 2.12-1.41 1.41L12 15.41l-2.12 2.12-1.41-1.41L10.59 14l-2.13-2.12 1.41-1.41L12 12.59ZM6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12ZM16 9H8v10h8V9Z"
};
var done = {
  name: "done",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m8.8 15.9-4.2-4.2-1.4 1.4 5.6 5.6 12-12-1.4-1.4L8.8 15.9Z"
};
var done_all = {
  name: "done_all",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m22.205 5.295-10.58 10.58-4.18-4.17-1.41 1.41 5.59 5.59 12-12-1.42-1.41Zm-4.24 1.41-1.41-1.41-6.34 6.34 1.41 1.41 6.34-6.34Zm-12 12-5.59-5.59 1.42-1.41 5.58 5.59-1.41 1.41Z"
};
var restore_from_trash = {
  name: "restore_from_trash",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m14.5 3 1 1H19v2H5V4h3.5l1-1h5ZM8 21c-1.1 0-2-.9-2-2V7h12v12c0 1.1-.9 2-2 2H8Zm0-7V9h8v5l-4-4-4 4Zm0 0v5h8v-5h-2v4h-4v-4H8Z"
};
var close_circle_outlined = {
  name: "close_circle_outlined",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.47 2 2 6.47 2 12s4.47 10 10 10 10-4.47 10-10S17.53 2 12 2Zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8Zm0-9.41L15.59 7 17 8.41 13.41 12 17 15.59 15.59 17 12 13.41 8.41 17 7 15.59 10.59 12 7 8.41 8.41 7 12 10.59Z"
};
var check = {
  name: "check",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m8.795 15.875-4.17-4.17-1.42 1.41 5.59 5.59 12-12-1.41-1.41-10.59 10.58Z"
};
var radio_button_selected = {
  name: "radio_button_selected",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2Zm0 18c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8Zm-5-8a5 5 0 1 1 10 0 5 5 0 0 1-10 0Z"
};
var radio_button_unselected = {
  name: "radio_button_unselected",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10-4.48 10-10 10S2 17.52 2 12Zm2 0c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8-8 3.58-8 8Z"
};
var switch_off = {
  name: "switch_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 7h10c2.76 0 5 2.24 5 5s-2.24 5-5 5H7c-2.76 0-5-2.24-5-5s2.24-5 5-5Zm-3 5c0 1.66 1.34 3 3 3s3-1.34 3-3-1.34-3-3-3-3 1.34-3 3Z"
};
var switch_on = {
  name: "switch_on",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 7h10c2.76 0 5 2.24 5 5s-2.24 5-5 5H7c-2.76 0-5-2.24-5-5s2.24-5 5-5Zm7 5c0 1.66 1.34 3 3 3s3-1.34 3-3-1.34-3-3-3-3 1.34-3 3Z"
};
var log_out = {
  name: "log_out",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5.988 20h7.95v2h-7.95A2 2 0 0 1 4 20V4a2 2 0 0 1 1.988-2h7.95v2h-7.95v16Zm8.745-2.7-1.49-1.4 2.955-2.9h-6.83v-2h6.856l-2.982-3 1.391-1.4L20 12l-5.267 5.3Z"
};
var log_in = {
  name: "log_in",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17.95 4H10V2h7.95a2 2 0 0 1 1.988 2v16a2 2 0 0 1-1.988 2H10v-2h7.95V4ZM9.366 17.3l-1.491-1.4 2.956-2.9H4v-2h6.857L7.875 8l1.391-1.4 5.367 5.4-5.267 5.3Z"
};
var check_circle_outlined = {
  name: "check_circle_outlined",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2Zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8Zm-2-5.83 6.59-6.59L18 9l-8 8-4-4 1.41-1.41L10 14.17Z",
  sizes: {
    small: {
      name: "check_circle_outlined_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M9 1C4.584 1 1 4.584 1 9s3.584 8 8 8 8-3.584 8-8-3.584-8-8-8Zm0 14c-3.308 0-6-2.693-6-6 0-3.308 2.692-6 6-6 3.307 0 6 2.692 6 6 0 3.307-2.693 6-6 6Zm-1.599-4.264 5.272-5.272L13.801 6.6l-6.4 6.4-3.2-3.2L5.33 8.672l2.072 2.064Z"
    }
  }
};
var close = {
  name: "close",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41Z",
  sizes: {
    small: {
      name: "close_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M14 5.007 12.993 4 9 7.993 5.007 4 4 5.007 7.993 9 4 12.993 5.007 14 9 10.007 12.993 14 14 12.993 10.007 9 14 5.007Z"
    }
  }
};
var search = {
  name: "search",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14.966 14.255h.79l4.99 5-1.49 1.49-5-4.99v-.79l-.27-.28a6.471 6.471 0 0 1-4.23 1.57 6.5 6.5 0 1 1 6.5-6.5c0 1.61-.59 3.09-1.57 4.23l.28.27Zm-9.71-4.5c0 2.49 2.01 4.5 4.5 4.5s4.5-2.01 4.5-4.5-2.01-4.5-4.5-4.5-4.5 2.01-4.5 4.5Z",
  sizes: {
    small: {
      name: "search_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M11.224 10.691h.592l3.743 3.75-1.118 1.118-3.75-3.743v-.592l-.202-.21a4.853 4.853 0 0 1-3.173 1.177 4.875 4.875 0 1 1 4.875-4.875 4.853 4.853 0 0 1-1.177 3.173l.21.202ZM3.94 7.316a3.37 3.37 0 0 0 3.375 3.375 3.37 3.37 0 0 0 3.375-3.375 3.37 3.37 0 0 0-3.375-3.375 3.37 3.37 0 0 0-3.375 3.375Z"
    }
  }
};
var checkbox = {
  name: "checkbox",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 3h14c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H5c-1.1 0-2-.9-2-2V5c0-1.1.9-2 2-2Zm4.3 13.29c.39.39 1.02.39 1.41 0l7.58-7.59a.996.996 0 1 0-1.41-1.41L10 14.17l-2.88-2.88a.996.996 0 1 0-1.41 1.41l3.59 3.59Z",
  sizes: {
    small: {
      name: "checkbox_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M3.75 2h10.5c.825 0 1.75.925 1.75 1.75v10.5c0 .825-.925 1.75-1.75 1.75H3.75c-.825 0-1.662-.925-1.662-1.75V3.75c0-.825.837-1.75 1.662-1.75Zm3.225 10.217a.747.747 0 0 0 1.057 0l5.685-5.692a.747.747 0 1 0-1.057-1.058l-5.16 5.16-2.16-2.16a.747.747 0 1 0-1.058 1.058l2.693 2.692Z"
    }
  }
};
var checkbox_outline = {
  name: "checkbox_outline",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 3h14c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H5c-1.1 0-2-.9-2-2V5c0-1.1.9-2 2-2Zm1 16h12c.55 0 1-.45 1-1V6c0-.55-.45-1-1-1H6c-.55 0-1 .45-1 1v12c0 .55.45 1 1 1Z",
  sizes: {
    small: {
      name: "checkbox_outline_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M3.5 2H14c.825 0 2 .925 2 1.75v10.5c0 .825-1.175 1.75-2 1.75H3.5c-.825 0-1.5-.925-1.5-1.75V3.75C2 2.925 2.675 2 3.5 2Zm1.214 12h8.572a.716.716 0 0 0 .714-.714V4.714A.716.716 0 0 0 13.286 4H4.714A.716.716 0 0 0 4 4.714v8.572c0 .393.321.714.714.714Z"
    }
  }
};
var checkbox_indeterminate = {
  name: "checkbox_indeterminate",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 3h14c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H5c-1.1 0-2-.9-2-2V5c0-1.1.9-2 2-2Zm3 10h8c.55 0 1-.45 1-1s-.45-1-1-1H8c-.55 0-1 .45-1 1s.45 1 1 1Z",
  sizes: {
    small: {
      name: "checkbox_indeterminate_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M3.5 2H14c.825 0 2 .925 2 1.75V14c0 .825-1.175 2-2 2H4c-.825 0-2-1.175-2-2V3.75C2 2.925 2.675 2 3.5 2Zm2.3 8h6.4c.44 0 .8-.45.8-1s-.36-1-.8-1H5.8c-.44 0-.8.45-.8 1s.36 1 .8 1Z"
    }
  }
};
var zip_file = {
  name: "zip_file",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2Zm-.83 2 2 2H13v2h2v2h-2v2h2v2h-2v2H4V6h5.17ZM15 16h2v-2h-2v-2h2v-2h-2V8h5v10h-5v-2Z"
};
var approve = {
  name: "approve",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m15.33 1.781 2.137 2.968 3.72.852-.377 3.58 2.541 2.826-2.54 2.765.374 3.56-3.705.91-2.082 2.976-3.393-1.33-3.301 1.33-2.167-2.966-3.722-.854.375-3.625-2.542-2.766L3.19 9.18 2.813 5.6l3.712-.85 2.076-2.97 3.395 1.33 3.334-1.33ZM9.338 4.218l-1.615 2.31-2.735.627.284 2.702-1.92 2.135 1.92 2.09-.285 2.762 2.724.625 1.69 2.312 2.594-1.045 2.667 1.045 1.61-2.302 2.741-.673-.286-2.723 1.921-2.09-1.92-2.136.284-2.702-2.727-.626-1.665-2.311-2.617 1.045-2.665-1.045ZM16.59 7.58 10 14.17l-2.59-2.58L6 13l4 4 8-8-1.41-1.42Z"
};
var calendar_event = {
  name: "calendar_event",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 4h-1V2h-2v2H8V2H6v2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2Zm0 16H5V10h14v10ZM5 6v2h14V6H5Zm2 6h10v2H7v-2Zm7 4H7v2h7v-2Z"
};
var calendar_accept = {
  name: "calendar_accept",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18 4h1c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H5c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2h1V2h2v2h8V2h2v2ZM5 20h14V10H5v10ZM5 8V6h14v2H5Zm11.49 4.53-5.93 5.93-3.17-3.17 1.06-1.06 2.11 2.11 4.87-4.87 1.06 1.06Z"
};
var calendar_reject = {
  name: "calendar_reject",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18 4h1c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H5c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2h1V2h2v2h8V2h2v2ZM5 20h14V10H5v10ZM5 8V6h14v2H5Zm4.29 10.47-1.06-1.06 2.44-2.44-2.44-2.44 1.06-1.06 2.44 2.44 2.44-2.44 1.06 1.06-2.44 2.44 2.44 2.44-1.06 1.06-2.44-2.44-2.44 2.44Z"
};
var timer = {
  name: "timer",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M9 1.505h6v2H9v-2Zm2 13v-6h2v6h-2Zm8.03-6.62 1.42-1.42c-.43-.51-.9-.99-1.41-1.41l-1.42 1.42A8.962 8.962 0 0 0 12 4.495a9 9 0 0 0-9 9c0 4.97 4.02 9 9 9s9-4.03 9-9c0-2.11-.74-4.06-1.97-5.61ZM5 13.505c0 3.87 3.13 7 7 7s7-3.13 7-7-3.13-7-7-7-7 3.13-7 7Z"
};
var timer_off = {
  name: "timer_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M9.625 1h6v2h-6V1Zm2 7v.86l2 2V8h-2Zm8 5c0-3.87-3.13-7-7-7-1.12 0-2.18.27-3.12.74l-1.47-1.47c1.34-.8 2.91-1.27 4.59-1.27 2.12 0 4.07.74 5.62 1.98l1.42-1.42c.51.42.98.9 1.41 1.41l-1.42 1.42a8.963 8.963 0 0 1 1.97 5.61c0 1.68-.47 3.25-1.27 4.59l-1.47-1.47c.47-.94.74-2 .74-3.12ZM3.785 3.86l-1.41 1.41 2.75 2.75a9.043 9.043 0 0 0-1.5 4.98c0 4.97 4.02 9 9 9 1.84 0 3.55-.55 4.98-1.5l2.5 2.5 1.41-1.41L3.785 3.86ZM5.625 13c0 3.87 3.13 7 7 7 1.29 0 2.49-.35 3.53-.95l-9.57-9.57a6.876 6.876 0 0 0-.96 3.52Z"
};
var calendar_today = {
  name: "calendar_today",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18 4h1c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h1V2h2v2h8V2h2v2ZM5 10v10h14V10H5Zm14-2H5V6h14v2Zm-7 4H7v5h5v-5Z"
};
var calendar_date_range = {
  name: "calendar_date_range",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 11h2v2H7v-2Zm14-5v14c0 1.1-.9 2-2 2H5a2 2 0 0 1-2-2l.01-14c0-1.1.88-2 1.99-2h1V2h2v2h8V2h2v2h1c1.1 0 2 .9 2 2ZM5 8h14V6H5v2Zm14 12V10H5v10h14Zm-4-7h2v-2h-2v2Zm-4 0h2v-2h-2v2Z"
};
var alarm = {
  name: "alarm",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7.945 3.441 6.664 1.905 2.057 5.75l1.28 1.536L7.946 3.44Zm9.392-1.535 4.608 3.843-1.282 1.536-4.607-3.843 1.281-1.536Zm-4.836 6.189H11v6l4.75 2.85.75-1.23-4-2.37v-5.25Zm-.5-4a9 9 0 1 0 .001 18.001 9 9 0 0 0-.001-18.001Zm-7 9c0 3.86 3.14 7 7 7s7-3.14 7-7-3.14-7-7-7-7 3.14-7 7Z"
};
var alarm_add = {
  name: "alarm_add",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m6.664 1.905 1.281 1.536-4.607 3.844-1.281-1.536 4.607-3.844Zm10.673 0 4.608 3.844-1.282 1.536-4.607-3.843 1.281-1.536ZM3.001 13.096a9 9 0 1 1 18.001.001 9 9 0 0 1-18.001-.001Zm9 7c-3.86 0-7-3.14-7-7s3.14-7 7-7 7 3.14 7 7-3.14 7-7 7Zm-1-8v-3h2v3h3v2h-3v3h-2v-3H8v-2h3Z"
};
var alarm_off = {
  name: "alarm_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m8.164 3.445-.46.38-1.42-1.42.6-.5 1.28 1.54Zm9.397-1.54 4.607 3.844-1.281 1.536-4.608-3.843 1.282-1.536Zm-7.297 4.48c.62-.18 1.28-.29 1.96-.29 3.86 0 7 3.14 7 7 0 .68-.11 1.34-.29 1.96l1.56 1.56c.47-1.08.73-2.27.73-3.52a9 9 0 0 0-12.53-8.28l1.57 1.57Zm-8.43-2.78 1.41-1.41 18.38 18.39-1.41 1.41-2.1-2.1a8.964 8.964 0 0 1-5.89 2.2 9 9 0 0 1-9-9c0-2.25.83-4.31 2.2-5.89l-.8-.8-1.06.88-1.28-1.54.92-.77-1.37-1.37Zm10.39 16.49c-3.86 0-7-3.14-7-7 0-1.7.61-3.26 1.62-4.47l9.85 9.85a6.956 6.956 0 0 1-4.47 1.62Z"
};
var alarm_on = {
  name: "alarm_on",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7.945 3.441 6.664 1.905 2.057 5.75l1.28 1.536L7.946 3.44Zm9.392-1.535 4.608 3.843-1.282 1.536-4.607-3.843 1.281-1.536Zm-6.796 12.719-2.13-2.13-1.06 1.06 3.18 3.18 6-6-1.06-1.06-4.93 4.95ZM12 4.095a9 9 0 1 0 .001 18.001 9 9 0 0 0-.001-18.001Zm-7 9c0 3.86 3.14 7 7 7s7-3.14 7-7-3.14-7-7-7-7 3.14-7 7Z"
};
var infinity = {
  name: "infinity",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18.6 6.62c-1.44 0-2.8.56-3.77 1.53L7.8 14.39c-.64.64-1.49.99-2.4.99-1.87 0-3.39-1.51-3.39-3.38 0-1.87 1.52-3.38 3.39-3.38.91 0 1.76.35 2.44 1.03l1.13 1 1.51-1.34L9.22 8.2A5.37 5.37 0 0 0 5.4 6.62C2.42 6.62 0 9.04 0 12s2.42 5.38 5.4 5.38c1.44 0 2.8-.56 3.77-1.53l7.03-6.24c.64-.64 1.49-.99 2.4-.99 1.87 0 3.39 1.51 3.39 3.38 0 1.87-1.52 3.38-3.39 3.38-.9 0-1.76-.35-2.44-1.03l-1.14-1.01-1.51 1.34 1.27 1.12a5.386 5.386 0 0 0 3.82 1.57c2.98 0 5.4-2.41 5.4-5.38 0-2.97-2.42-5.37-5.4-5.37Z"
};
var hourglass_empty = {
  name: "hourglass_empty",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M6 2v6h.01L6 8.01 10 12l-4 4 .01.01H6V22h12v-5.99h-.01L18 16l-4-4 4-3.99-.01-.01H18V2H6Zm10 14.5V20H8v-3.5l4-4 4 4ZM8 4v3.5l4 4 4-4V4H8Z"
};
var hourglass_full = {
  name: "hourglass_full",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M6 2v6h.01L6 8.01 10 12l-4 4 .01.01H6V22h12v-5.99h-.01L18 16l-4-4 4-3.99-.01-.01H18V2H6Z"
};
var calendar = {
  name: "calendar",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 4h-1V2h-2v2H8V2H6v2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2Zm0 16H5V10h14v10ZM5 6v2h14V6H5Z",
  sizes: {
    small: {
      name: "calendar_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M13.5 3H13V2h-1v1H6V2H5v1h-.5C3.5 3 3 3.5 3 4.6v9.1c0 .715.6 1.3 1.333 1.3h9.334C14.4 15 15 14.415 15 13.7V4.6c0-1.1-.5-1.6-1.5-1.6ZM13 13H5V8h8v5ZM5 5v1h8V5H5Z"
    }
  }
};
var time = {
  name: "time",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10S17.5 2 12 2Zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8ZM11 7h1.5v5.2l4.5 2.7-.8 1.3L11 13V7Z",
  sizes: {
    small: {
      name: "time_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M9 2C5.15 2 2 5.15 2 9s3.15 7 7 7 7-3.15 7-7-3.15-7-7-7Zm-.002 12.6a5.607 5.607 0 0 1-5.6-5.6c0-3.087 2.513-5.6 5.6-5.6 3.087 0 5.6 2.513 5.6 5.6 0 3.087-2.513 5.6-5.6 5.6ZM8 6h1v3.391l3 1.761-.533.848L8 9.913V6Z"
    }
  }
};
var no_craning = {
  name: "no_craning",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 4.5 4.5 3 21 19.5 19.5 21 3 4.5ZM13.915 11H14a1 1 0 0 0 .832-.445l4-6A1 1 0 0 0 18 3H6c-.027 0-.055.001-.082.003L7.915 5h8.217l-2.667 4h-1.55l2 2Zm.976 6.805 1.531 1.531A5 5 0 0 1 7 17a1 1 0 1 1 2 0 3 3 0 0 0 5.891.805ZM13 7a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"
};
var craning = {
  name: "craning",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5.118 3.528A1 1 0 0 1 6 3h12a1 1 0 0 1 .832 1.555l-4 6A1 1 0 0 1 14 11h-1v1.1a5.002 5.002 0 0 1-1 9.9 5 5 0 0 1-5-5 1 1 0 1 1 2 0 3 3 0 1 0 3-3h-1v-3h-1a1 1 0 0 1-.832-.445l-4-6a1 1 0 0 1-.05-1.027ZM13.465 9H10.535L7.87 5h8.262l-2.666 4ZM12 8a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
};
var toolbox = {
  name: "toolbox",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M9 4a2 2 0 0 0-2 2v2H5a2 2 0 0 0-2 2v10h18V10a2 2 0 0 0-2-2h-2V6a2 2 0 0 0-2-2H9Zm10 9v-3H5v3h2v-1h2v1h6v-1h2v1h2Zm-4 2v1h2v-1h2v3H5v-3h2v1h2v-1h6Zm0-7V6H9v2h6Z"
};
var cable = {
  name: "cable",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 4.5a1 1 0 0 0-2 0v4a1 1 0 1 0 2 0v-4Zm10.25-.75A4.75 4.75 0 0 0 12.5 8.5v6.75a2.75 2.75 0 1 1-5.5 0V11.5a1 1 0 1 0-2 0v3.75a4.75 4.75 0 1 0 9.5 0V8.5a2.75 2.75 0 0 1 5.5 0V19a1 1 0 0 0 2 0V8.5a4.75 4.75 0 0 0-4.75-4.75ZM1.293 7.793a1 1 0 0 1 1.414 0l1.414 1.414a1 1 0 1 1-1.414 1.414L1.293 9.207a1 1 0 0 1 0-1.414Zm8 0a1 1 0 0 1 1.414 1.414l-1.414 1.414a1 1 0 0 1-1.415-1.414l1.415-1.414Z"
};
var beat = {
  name: "beat",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11.5 4a1 1 0 0 0-.97.757l-1.845 7.378-.79-1.582a1 1 0 0 0-1.79 0L4.383 14H2a1 1 0 1 0 0 2h3a1 1 0 0 0 .894-.553L7 13.237l1.106 2.21a1 1 0 0 0 1.864-.204l1.53-6.12 2.53 10.12a1 1 0 0 0 1.94 0l1.03-4.12.03.12A1 1 0 0 0 18 16h4a1 1 0 1 0 0-2h-3.22l-.81-3.242a1 1 0 0 0-1.94 0L15 14.877l-2.53-10.12A1 1 0 0 0 11.5 4Z"
};
var gas = {
  name: "gas",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M13.205 13.636c.985 1.83 1.238 4.036 1.127 5.806-.093 1.49-.444 2.67-.832 3.058.75-.346 1.758-1.11 2.757-2.149C18.146 18.387 20 15.444 20 12.5c0-4.216-3.82-7.257-6.73-9.573-.186-.15-.37-.295-.548-.438C12.056 1.954 11.46 1.46 11 1c0 .645-.324 1.29-.832 1.96C9.672 3.615 9 4.292 8.284 5.016 6.316 7 4 9.336 4 12.5c0 3.012 1.942 5.8 3.876 7.765.955.97 1.908 1.74 2.624 2.235-.294-.765-.599-1.893-.727-3.175-.178-1.79-.013-3.882 1.004-5.707A7.33 7.33 0 0 1 12 12l.001.001a6.984 6.984 0 0 1 1.204 1.635ZM7.75 17.044c.17-2.146.909-4.532 2.835-6.458L12 9.172l1.414 1.414c1.875 1.875 2.641 4.4 2.866 6.583C17.325 15.65 18 13.996 18 12.5c0-1.919-1.131-3.674-2.953-5.438-.892-.863-1.87-1.652-2.832-2.42l-.217-.172-.29-.23c-.089.114-.178.224-.264.328-.518.623-1.186 1.297-1.811 1.927-.242.244-.477.482-.694.707C7.186 9.018 6 10.624 6 12.5c0 1.5.68 3.065 1.751 4.544Z"
};
var gear = {
  name: "gear",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11.424 2a.5.5 0 0 0-.493.418l-.224 1.345a.52.52 0 0 1-.408.418 7.955 7.955 0 0 0-2.646 1.102.52.52 0 0 1-.586-.006L5.952 4.48a.5.5 0 0 0-.644.053l-.816.816a.5.5 0 0 0-.053.644l.804 1.126a.52.52 0 0 1 .01.582 7.953 7.953 0 0 0-1.079 2.63.52.52 0 0 1-.419.41l-1.378.23a.5.5 0 0 0-.418.494v1.153a.5.5 0 0 0 .418.493l1.397.233a.52.52 0 0 1 .418.405c.209.936.581 1.81 1.086 2.59a.52.52 0 0 1-.007.586L4.44 18.09a.5.5 0 0 0 .053.644l.816.815a.5.5 0 0 0 .644.053l1.176-.84a.52.52 0 0 1 .582-.009c.777.495 1.646.86 2.575 1.063a.52.52 0 0 1 .407.418l.239 1.43a.5.5 0 0 0 .493.419h1.153a.5.5 0 0 0 .493-.418l.238-1.43a.52.52 0 0 1 .407-.42 7.952 7.952 0 0 0 2.576-1.062.52.52 0 0 1 .582.01l1.176.84a.5.5 0 0 0 .644-.054l.815-.815a.5.5 0 0 0 .053-.644l-.832-1.165a.52.52 0 0 1-.007-.585 7.955 7.955 0 0 0 1.087-2.591.52.52 0 0 1 .418-.405l1.397-.233a.5.5 0 0 0 .418-.493v-1.153a.5.5 0 0 0-.418-.493l-1.379-.23a.52.52 0 0 1-.419-.41 7.954 7.954 0 0 0-1.078-2.63.52.52 0 0 1 .01-.583l.803-1.126a.5.5 0 0 0-.053-.644l-.815-.816a.5.5 0 0 0-.644-.053l-1.116.797a.52.52 0 0 1-.585.006 7.956 7.956 0 0 0-2.646-1.102.52.52 0 0 1-.408-.418l-.224-1.345A.5.5 0 0 0 12.577 2h-1.153ZM18 12a6 6 0 1 1-12 0 6 6 0 0 1 12 0Zm-4 0a2 2 0 1 1-4 0 2 2 0 0 1 4 0Zm2 0a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z"
};
var bearing = {
  name: "bearing",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10Zm0-2a8 8 0 1 0 0-16 8 8 0 0 0 0 16Zm0-4a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm0-2a2 2 0 1 0 0-4 2 2 0 0 0 0 4Zm1-8a1 1 0 1 1-2 0 1 1 0 0 1 2 0Zm4 2a1 1 0 1 1-2 0 1 1 0 0 1 2 0Zm0 8a1 1 0 1 1-2 0 1 1 0 0 1 2 0Zm-8 0a1 1 0 1 1-2 0 1 1 0 0 1 2 0Zm0-8a1 1 0 1 1-2 0 1 1 0 0 1 2 0Zm10 4a1 1 0 1 1-2 0 1 1 0 0 1 2 0ZM7 12a1 1 0 1 1-2 0 1 1 0 0 1 2 0Zm6 6a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"
};
var pressure = {
  name: "pressure",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16.046 2.867c.101.045.144.164.097.264l-.68 1.448a.204.204 0 0 1-.268.098 8 8 0 0 0-6.448.003.204.204 0 0 1-.268-.098l-.681-1.448a.197.197 0 0 1 .096-.264 10 10 0 0 1 8.152-.003Zm5.85 7.894a.197.197 0 0 1-.176.22l-1.591.166a.204.204 0 0 1-.223-.178 8 8 0 0 0-3.337-5.517.204.204 0 0 1-.055-.28L17.4 3.84a.197.197 0 0 1 .276-.054 9.999 9.999 0 0 1 4.22 6.975Zm-19.67.215a.197.197 0 0 1-.175-.22 10 10 0 0 1 4.223-6.973.197.197 0 0 1 .276.054l.885 1.333a.204.204 0 0 1-.054.28 8 8 0 0 0-3.34 5.515.204.204 0 0 1-.223.178l-1.591-.167ZM6 20h-.027a10.001 10.001 0 0 1-3.972-7.26.197.197 0 0 1 .186-.21l1.598-.087a.204.204 0 0 1 .213.19A8 8 0 0 0 8.1 19h7.746a8 8 0 0 0 4.102-6.368.204.204 0 0 1 .213-.189l1.598.087c.11.006.195.1.187.21A10.001 10.001 0 0 1 17.973 20h-.027a.197.197 0 0 1-.053.06c-.478.35-.986.659-1.52.92a.198.198 0 0 1-.087.02H7.66a.198.198 0 0 1-.087-.02 9.998 9.998 0 0 1-1.52-.92A.195.195 0 0 1 6 20Zm5.811-12.964a.5.5 0 0 0-.388.416l-.693 4.776a2 2 0 1 0 2.486 0l-.693-4.776a.5.5 0 0 0-.388-.416l-.055-.012a.5.5 0 0 0-.214 0l-.055.012Z"
};
var platform = {
  name: "platform",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 8h2v5H7V8Zm1-5v3H6a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1h4a1 1 0 0 0 1-1V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1Zm12 15v-2H4v2h3v4h2v-4h6v4h2v-4h3Zm-6-5h3v-1.955l-3-1.143V13Zm-2 1V8.451a1 1 0 0 1 1.356-.934l5 1.904a1 1 0 0 1 .644.935V14a1 1 0 0 1-1 1h-5a1 1 0 0 1-1-1Z"
};
var circuit = {
  name: "circuit",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2a1 1 0 0 0-1 1v4a1 1 0 0 0 2 0V3a1 1 0 0 0-1-1ZM4.707 5.707a1 1 0 0 0 0 1.414l3.847 3.847a4.002 4.002 0 0 0 2.454 5.908A.998.998 0 0 0 11 17v4a1 1 0 0 0 2 0v-4a.998.998 0 0 0-.008-.124 4.002 4.002 0 1 0-3.024-7.322L6.12 5.707a1 1 0 0 0-1.414 0ZM14 13a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z"
};
var engineering = {
  name: "engineering",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M9.045 15c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4Zm-6 4c.22-.72 3.31-2 6-2 2.7 0 5.8 1.29 6 2h-12ZM4.785 9h.26c0 2.21 1.79 4 4 4s4-1.79 4-4h.26c.27 0 .49-.22.49-.49v-.02a.49.49 0 0 0-.49-.49h-.26c0-1.48-.81-2.75-2-3.45v.95c0 .28-.22.5-.5.5s-.5-.22-.5-.5V4.14a4.09 4.09 0 0 0-1-.14c-.35 0-.68.06-1 .14V5.5c0 .28-.22.5-.5.5s-.5-.22-.5-.5v-.95c-1.19.7-2 1.97-2 3.45h-.26a.49.49 0 0 0-.49.49v.03c0 .26.22.48.49.48Zm6.26 0c0 1.1-.9 2-2 2s-2-.9-2-2h4ZM22.025 6.23l.93-.83-.75-1.3-1.19.39c-.14-.11-.3-.2-.47-.27L20.295 3h-1.5l-.25 1.22c-.17.07-.33.16-.48.27l-1.18-.39-.75 1.3.93.83c-.02.17-.02.35 0 .52l-.93.85.75 1.3 1.2-.38c.13.1.28.18.43.25l.28 1.23h1.5l.27-1.22c.16-.07.3-.15.44-.25l1.19.38.75-1.3-.93-.85c.03-.19.02-.36.01-.53Zm-2.48 1.52a1.25 1.25 0 1 1 0-2.5 1.25 1.25 0 0 1 0 2.5ZM19.445 10.79l-.85.28c-.1-.08-.21-.14-.33-.19l-.18-.88h-1.07l-.18.87c-.12.05-.24.12-.34.19l-.84-.28-.54.93.66.59c-.01.13-.01.25 0 .37l-.66.61.54.93.86-.27c.1.07.2.13.31.18l.18.88h1.07l.19-.87c.11-.05.22-.11.32-.18l.85.27.54-.93-.66-.61c.01-.13.01-.25 0-.37l.66-.59-.53-.93Zm-1.9 2.6c-.49 0-.89-.4-.89-.89s.4-.89.89-.89.89.4.89.89-.4.89-.89.89Z"
};
var ducting = {
  name: "ducting",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M8 17c-.6 0-1-.5-1-1V4c0-.5.4-1 1-1s1 .5 1 1v1.1h9c2.2 0 4 1.8 4 4v9.007c.465.06.9.526.9.993 0 .566-.356.954-.9.996v.004h-9.1c-.5 0-1-.4-1-1s.4-1 1-1h.1v-3H9v.9c0 .5-.5 1-1 1Zm12 1.1h-5v-3c0-1.1-.9-2-2-2H9v-6h9c1.1 0 2 .9 2 2v9ZM2.9 6.7c.4-.4.9-.6 1.4-.6.938 0 1.613.705 1.775.874L6.1 7 4.6 8.3c-.1-.1-.2-.2-.3-.2-.4.5-.9.6-1.3.6h-.2c-1-.1-1.7-1-1.8-1.2l1.7-1.1c0 .1.1.2.2.3Zm0 2.9c.4-.4.9-.6 1.4-.6.938 0 1.613.705 1.775.874L6.1 9.9l-1.5 1.3c-.1-.1-.2-.2-.3-.2-.4.5-.9.6-1.3.6h-.2c-1-.1-1.7-1-1.8-1.2l1.7-1.2c0 .1.1.3.2.4Zm1.4 2.2c-.5 0-1 .2-1.4.6-.1-.1-.2-.2-.2-.3L1 13.2c.1.2.8 1.1 1.8 1.2H3c.4 0 .9-.1 1.3-.6.1 0 .2.1.3.2l1.5-1.3-.025-.026c-.162-.17-.837-.874-1.775-.874Z"
};
var formula = {
  name: "formula",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M9 2C5.9 2 4.3 3.6 4.3 6.6v1.7H2v2.2h2.3V22h3V10.5h3.3V8.3H7.3V7c0-1.7.7-2.6 2.3-2.6.5 0 1 0 1.7.2V2.3C10.6 2.1 9.8 2 9 2Zm9.7 18.4-1.5-2.9 1.2-2.4h-1.9l-.6 1.2-.5-1.2h-2.3l1.4 2.7-1.5 2.6h2l.7-1.4.7 1.4h2.3ZM12 22.8c-3.9-4.8-1.7-9.6 0-11.5l1.5 1.3-.7-.6.7.7c-.2.2-3.7 4.2 0 8.9L12 22.8Zm6.3-1.2 1.5 1.3c1.7-1.9 3.9-6.7.1-11.5l-1.6 1.3c3.7 4.7.2 8.7 0 8.9Z"
};
var manual_valve = {
  name: "manual_valve",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M13 4h2c.5 0 1-.4 1-1s-.4-1-1-1H9c-.6 0-1 .4-1 1s.5 1 1 1h2v7.464L2 4.1v18l10.061-8.232L22 22V4l-9 7.364V4Zm7 13.9V8.3L14.1 13l5.9 4.9ZM9.8 13.1 4 8.3v9.5l5.8-4.7Z"
};
var pipe_support = {
  name: "pipe_support",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16.4 4.8c-.2-.2-.4-.5-.4-.8a1 1 0 0 1 1.6-.8c2.1 1.7 3.4 4.2 3.4 7.1a9.012 9.012 0 0 1-4 7.509V20.2c0 .6-.4 1-1 1H8c-.6 0-1-.4-1-1v-2.56C4.58 15.92 2.936 13.062 3 10c.1-2.7 1.3-5.1 3.3-6.7.6-.5 1.6 0 1.6.8 0 .3-.2.6-.4.8C6 6.2 5 8.1 5 10.3c0 4.3 3.9 7.7 8.4 6.8 2.8-.5 5-2.7 5.5-5.5.5-2.7-.6-5.3-2.5-6.8Z"
};
var heat_trace = {
  name: "heat_trace",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5.34 22.061c-2.01 0-3.3-1.38-3.3-3.5 0-1.03-.03-10.66-.04-13.5 0-.55.44-1 1-1 .55 0 1 .44 1 .99.01 2.84.04 12.47.04 13.5 0 1.5.95 1.5 1.3 1.5 1.08 0 1.17-1.42 1.17-1.43v-1.15c0-3.04-.01-11.11 0-11.92.02-.78.32-1.88 1.13-2.66.63-.6 1.43-.9 2.35-.88 2.17.05 2.99 2.16 3.01 3.57v3.68c-.01 3.33-.01 8.92 0 9.29.07 1.31.7 1.5 1.3 1.5h.02c.64 0 1.1-.26 1.25-1.47l.01-.11c.03-.5 0-8.74-.01-11.44v-1.45c0-1.41.86-3.53 3.23-3.58.79-.02 1.51.26 2.06.8.9.88 1.14 2.22 1.14 2.8v13.47c0 .55-.45 1-1 1s-1-.45-1-1V5.601c0-.19-.13-.97-.53-1.36-.17-.16-.34-.25-.62-.23-1.23.03-1.27 1.42-1.27 1.58v1.44c.04 10.56.02 11.55-.02 11.75-.37 2.99-2.41 3.29-3.28 3.28-1.94-.02-3.17-1.29-3.28-3.4-.02-.35-.01-4.29 0-9.4v-3.64c0-.02-.05-1.59-1.06-1.61-.38-.02-.67.1-.91.32-.34.33-.51.89-.52 1.26-.02.79-.01 8.85 0 11.88v1.16c0 1.43-.98 3.43-3.17 3.43Z"
};
var instrument = {
  name: "instrument",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 11c0-3.9-3.1-7-7-7s-7 3.1-7 7c0 1.818.673 3.461 1.787 4.699a9.117 9.117 0 0 1 4.355-1.65L15 7.2c.3-.5.9-.7 1.4-.4.5.3.7.9.4 1.4l-3.33 5.912a9.22 9.22 0 0 1 3.777 1.548A6.985 6.985 0 0 0 19 11Zm-3.36 5.997C14.618 16.352 13.36 16 12.1 16c-1.344 0-2.577.39-3.657 1.047A7.025 7.025 0 0 0 12 18a7.02 7.02 0 0 0 3.64-1.003ZM3 11c0-5 4-9 9-9s9 4 9 9a8.958 8.958 0 0 1-8 8.946V21c0 .6-.4 1-1 1s-1-.4-1-1v-1.054A8.958 8.958 0 0 1 3 11Z"
};
var junction_box = {
  name: "junction_box",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3.023 3.695A2 2 0 0 1 5 2h14a2 2 0 0 1 2 2v13a2 2 0 0 1-2 2h-2v3h-2v-3h-2v3h-2v-3H9v3H7v-3H5a2 2 0 0 1-2-2V4c0-.104.008-.205.023-.305ZM5 4v13h14V4H5Z"
};
var line = {
  name: "line",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2h10a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-2a2 2 0 0 0-2 2H7a2 2 0 0 0-2-2H3Zm4 4v4h10v-4H7ZM5 8H3v8h2V8Zm16 8h-2V8h2v8Z"
};
var telecom = {
  name: "telecom",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m13.168 1-.648 1.794c.057.019 5.96 2.203 8.716 8.939L23 11.008C19.92 3.5 13.435 1.095 13.168 1Zm-.82 5.104.495-1.841c.182.047 4.454 1.23 6.876 6.716l-1.745.773c-2.04-4.608-5.483-5.61-5.626-5.648Zm4.11 5.505-1.802.63c-.85-2.424-2.546-2.738-2.623-2.748l.277-1.889c.135.02 2.901.446 4.148 4.007Zm-3.49 2.614c0 .391-.086.754-.22 1.097l4.244 4.16a1.181 1.181 0 0 1-.153 1.793c-1.583 1.126-3.585 1.718-5.645 1.718-2.556 0-5.197-.916-7.19-2.872C.408 16.6.16 10.903 2.62 7.316c.229-.334.6-.515.972-.515.296 0 .582.105.811.334l4.253 4.178a3.024 3.024 0 0 1 1.173-.229 3.14 3.14 0 0 1 3.138 3.14Zm-7.63 4.541c1.526 1.498 3.605 2.328 5.856 2.328 1.363 0 2.68-.324 3.776-.906l-3.414-3.349-1.459-1.44c.544-.134.954-.601.954-1.183 0-.678-.553-1.231-1.23-1.231-.592 0-1.059.43-1.183.983l-.858-.85-4.034-3.959c-1.364 2.729-1.182 6.888 1.593 9.607Z"
};
var toolbox_wheel = {
  name: "toolbox_wheel",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 3a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v2h2a2 2 0 0 1 2 2v10H3V7a2 2 0 0 1 2-2h2V3Zm12 4v3h-2V9h-2v1H9V9H7v1H5V7h14Zm-4 6v-1H9v1H7v-1H5v3h14v-3h-2v1h-2Zm0-10v2H9V3h6ZM2 18h20v2H2v-2Zm5.5 2a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3Zm9 0a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3Z"
};
var toolbox_rope = {
  name: "toolbox_rope",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 1h20v2h-9l3 3h-2l-2-2-2 2H8l3-3H2V1Zm5 8a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v2h2a2 2 0 0 1 2 2v10H3V13a2 2 0 0 1 2-2h2V9Zm12 4v3h-2v-1h-2v1H9v-1H7v1H5v-3h14Zm-4 6v-1H9v1H7v-1H5v3h14v-3h-2v1h-2Zm0-10v2H9V9h6Z"
};
var oil = {
  name: "oil",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16.201 19.117C15.016 20.404 13.654 21 12 21c-1.654 0-3.016-.596-4.201-1.883C6.606 17.82 6 16.264 6 14.328c0-1.566.397-2.784 1.108-3.76.805-1.105 2.427-3.209 4.883-6.328 2.464 3.162 4.098 5.286 4.913 6.383.699.942 1.096 2.138 1.096 3.706 0 1.935-.606 3.49-1.799 4.788ZM5.491 9.39c.861-1.183 2.606-3.442 5.233-6.775L12 1l1.267 1.623C15.899 6 17.646 8.27 18.509 9.43 19.503 10.769 20 12.4 20 14.329c0 2.408-.776 4.456-2.327 6.142C16.12 22.157 14.23 23 12 23s-4.121-.843-5.673-2.53C4.776 18.786 4 16.738 4 14.33c0-1.928.497-3.574 1.49-4.938ZM8.5 13a.5.5 0 0 0-1 0c0 1.233.127 2.437.604 3.54.482 1.114 1.306 2.09 2.639 2.889a.5.5 0 1 0 .514-.858c-1.167-.7-1.843-1.52-2.236-2.429-.398-.92-.521-1.967-.521-3.142Z"
};
var oil_barrel = {
  name: "oil_barrel",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 3a1 1 0 0 1 1-1h18a1 1 0 1 1 0 2h-2v7h1a1 1 0 1 1 0 2h-1v7h2a1 1 0 1 1 0 2H3a1 1 0 1 1 0-2h2v-7H4a1 1 0 1 1 0-2h1V4H3a1 1 0 0 1-1-1Zm15 10v7H7v-7a1 1 0 1 0 0-2V4h10v7a1 1 0 1 0 0 2Zm-5 3c.836 0 1.546-.307 2.127-.92.582-.613.873-1.357.873-2.233 0-.701-.186-1.295-.56-1.781A206.829 206.829 0 0 0 12 8c-1.255 1.538-2.068 2.555-2.44 3.051-.374.496-.56 1.095-.56 1.796 0 .876.29 1.62.873 2.233.582.613 1.29.92 2.127.92Z"
};
var wellbore = {
  name: "wellbore",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M13 3h-2v10.083c0 .507.448.917 1 .917s1-.41 1-.917V3ZM7 8H3V6h6v15H7V8Zm10 0h4V6h-6v15h2V8Z"
};
var share = {
  name: "share",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16.04 16.89c.52-.47 1.2-.77 1.96-.77 1.61 0 2.92 1.31 2.92 2.92 0 1.61-1.31 2.92-2.92 2.92-1.61 0-2.92-1.31-2.92-2.92 0-.22.03-.44.08-.65l-7.12-4.16c-.54.5-1.25.81-2.04.81-1.66 0-3-1.34-3-3s1.34-3 3-3c.79 0 1.5.31 2.04.81l7.05-4.11c-.05-.23-.09-.46-.09-.7 0-1.66 1.34-3 3-3s3 1.34 3 3-1.34 3-3 3c-.79 0-1.5-.31-2.04-.81l-7.05 4.11c.05.23.09.46.09.7 0 .24-.04.47-.09.7l7.13 4.15ZM19 5.04c0-.55-.45-1-1-1s-1 .45-1 1 .45 1 1 1 1-.45 1-1Zm-13 8c-.55 0-1-.45-1-1s.45-1 1-1 1 .45 1 1-.45 1-1 1Zm11 6.02c0 .55.45 1 1 1s1-.45 1-1-.45-1-1-1-1 .45-1 1Z"
};
var skype = {
  name: "skype",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7.5 2A5.5 5.5 0 0 0 2 7.5c0 1.231.409 2.362 1.092 3.28-.054.4-.092.805-.092 1.22a9 9 0 0 0 9 9c.415 0 .82-.038 1.22-.092A5.468 5.468 0 0 0 16.5 22a5.5 5.5 0 0 0 5.5-5.5 5.468 5.468 0 0 0-1.092-3.28c.054-.4.092-.805.092-1.22a9 9 0 0 0-9-9c-.415 0-.82.038-1.22.092A5.468 5.468 0 0 0 7.5 2Zm0 2c.753 0 1.475.24 2.086.695l.652.489.809-.11C11.42 5.024 11.724 5 12 5c3.86 0 7 3.14 7 7 0 .276-.023.578-.074.953l-.11.809.489.652A3.47 3.47 0 0 1 20 16.5c0 1.93-1.57 3.5-3.5 3.5a3.47 3.47 0 0 1-2.086-.695l-.652-.489-.809.11c-.375.05-.677.074-.953.074-3.86 0-7-3.14-7-7 0-.276.023-.578.074-.953l.11-.809-.489-.652A3.471 3.471 0 0 1 4 7.5C4 5.57 5.57 4 7.5 4Zm4.35 3C8.09 7 7.813 9.209 7.813 9.754c0 1.084.621 1.926 1.683 2.334 1.13.434 2.5.694 3.277.906 1.217.333 1.063 1.004 1.063 1.102 0 1.03-1.41 1.295-2.012 1.238-.68-.064-1.369.13-2.086-1.346-.128-.264-.363-.841-1.07-.841-.306 0-1.07.232-1.07.962 0 1.432 1.599 2.891 4.38 2.891 1.454 0 4.167-.503 4.168-3.111 0-2.271-2.29-2.67-3.974-3.104-.461-.119-2.18-.309-2.18-1.209 0-.267.374-.908 1.72-.908 1.388 0 1.69.65 1.874.986.154.267.28.474.435.573a1.091 1.091 0 0 0 1.375-.116c.2-.207.3-.435.3-.695C15.695 8.285 14.357 7 11.85 7Z"
};
var whats_app = {
  name: "whats_app",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12.012 2c-5.506 0-9.989 4.478-9.99 9.984a9.964 9.964 0 0 0 1.333 4.993L2 22l5.232-1.236a9.981 9.981 0 0 0 4.774 1.215h.004c5.505 0 9.985-4.48 9.988-9.985a9.922 9.922 0 0 0-2.922-7.066A9.923 9.923 0 0 0 12.012 2Zm-.002 2a7.95 7.95 0 0 1 5.652 2.342 7.929 7.929 0 0 1 2.336 5.65c-.002 4.404-3.584 7.986-7.99 7.986a7.999 7.999 0 0 1-3.817-.97l-.673-.367-.745.175-1.968.465.48-1.785.217-.8-.414-.72a7.98 7.98 0 0 1-1.067-3.992C4.023 7.582 7.607 4 12.01 4ZM8.477 7.375a.917.917 0 0 0-.666.313c-.23.248-.875.852-.875 2.08 0 1.228.894 2.415 1.02 2.582.123.166 1.726 2.765 4.263 3.765 2.108.831 2.536.667 2.994.625.458-.04 1.477-.602 1.685-1.185.208-.583.209-1.085.147-1.188-.062-.104-.229-.166-.479-.29-.249-.126-1.476-.728-1.705-.811-.229-.083-.396-.125-.562.125-.166.25-.643.81-.79.976-.145.167-.29.19-.54.065-.25-.126-1.054-.39-2.008-1.24-.742-.662-1.243-1.477-1.389-1.727-.145-.25-.013-.386.112-.51.112-.112.248-.291.373-.437.124-.146.167-.25.25-.416.083-.166.04-.313-.022-.438s-.547-1.357-.77-1.851c-.186-.415-.384-.425-.562-.432-.145-.006-.31-.006-.476-.006Z"
};
var facebook_messenger = {
  name: "facebook_messenger",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.486 2 2 6.262 2 11.5c0 2.545 1.088 4.988 3 6.771v4.346l4.08-2.039c.96.28 1.94.422 2.92.422 5.514 0 10-4.262 10-9.5S17.514 2 12 2Zm0 2c4.411 0 8 3.365 8 7.5S16.411 19 12 19a8.461 8.461 0 0 1-2.361-.342l-.752-.219-.701.35L7 19.383V17.402l-.637-.591C4.861 15.409 4 13.472 4 11.5 4 7.365 7.589 4 12 4Zm-1 5-5 5 4.5-2 2.5 2 5-5-4.5 2L11 9Z"
};
var microsoft_excel = {
  name: "microsoft_excel",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14 3 2 5v14l12 2v-2h7a1 1 0 0 0 1-1V6a1 1 0 0 0-1-1h-7V3Zm-2 2.361V18.64l-8-1.332V6.693l8-1.332ZM14 7h2v2h-2V7Zm4 0h2v2h-2V7ZM5.176 8.297l1.885 3.697L5 15.704h1.736l1.123-2.395c.075-.23.126-.4.15-.514h.016c.041.238.091.407.133.492l1.113 2.414H11l-1.994-3.734 1.937-3.67h-1.62l-1.03 2.197c-.1.285-.167.505-.201.647h-.026a4.519 4.519 0 0 0-.19-.63l-.923-2.214H5.176ZM14 11h2v2h-2v-2Zm4 0h2v2h-2v-2Zm-4 4h2v2h-2v-2Zm4 0h2v2h-2v-2Z"
};
var microsoft_word = {
  name: "microsoft_word",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14 3 2 5v14l12 2v-2h7a1 1 0 0 0 1-1V6a1 1 0 0 0-1-1h-7V3Zm-2 2.361V18.64l-8-1.332V6.693l8-1.332ZM14 7h6v2h-6V7ZM4.5 8.5l1.299 7h1.293l.847-3.742a5.68 5.68 0 0 0 .09-.783h.018c.008.288.031.549.072.783l.83 3.742h1.242l1.309-7h-1.365l-.451 3.223c-.033.27-.059.53-.067.783h-.015a7.206 7.206 0 0 0-.082-.748L8.967 8.5H7.2l-.594 3.268a5.285 5.285 0 0 0-.1.82h-.025a5.965 5.965 0 0 0-.064-.803L5.951 8.5H4.5ZM14 11h6v2h-6v-2Zm0 4h6v2h-6v-2Z"
};
var microsoft_powerpoint = {
  name: "microsoft_powerpoint",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14 3 2 5v14l12 2v-2h7a1 1 0 0 0 1-1V6a1 1 0 0 0-1-1h-7V3Zm-2 2.361V18.64l-8-1.332V6.693l8-1.332ZM14 7h6v10h-6v-2h4v-2h-4V7Zm0 3a2 2 0 1 0 4 0h-2V8a2 2 0 0 0-2 2ZM4.988 8v8H6.5v-2.969h1.545c.93 0 1.653-.216 2.17-.652.517-.437.773-1.047.773-1.832 0-.766-.263-1.384-.79-1.85C9.668 8.232 8.957 8 8.06 8H4.988ZM6.5 9.188h1.506c.436.005.777.128 1.025.365.25.236.375.55.375.945 0 .401-.123.708-.369.92-.246.212-.601.318-1.068.318H6.5V9.187Z"
};
var github = {
  name: "github",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M10.9 2.1c-4.6.5-8.3 4.2-8.8 8.7-.5 4.7 2.2 8.9 6.3 10.5.3.1.6-.1.6-.5v-1.6s-.4.1-.9.1c-1.4 0-2-1.2-2.1-1.9-.1-.4-.3-.7-.6-1-.3-.1-.4-.1-.4-.2 0-.2.3-.2.4-.2.6 0 1.1.7 1.3 1 .5.8 1.1 1 1.4 1 .4 0 .7-.1.9-.2.1-.7.4-1.4 1-1.8-2.3-.5-4-1.8-4-4 0-1.1.5-2.2 1.2-3-.1-.2-.2-.7-.2-1.4 0-.4 0-1 .3-1.6 0 0 1.4 0 2.8 1.3.5-.2 1.2-.3 1.9-.3s1.4.1 2 .3C15.3 6 16.8 6 16.8 6c.2.6.2 1.2.2 1.6 0 .8-.1 1.2-.2 1.4.7.8 1.2 1.8 1.2 3 0 2.2-1.7 3.5-4 4 .6.5 1 1.4 1 2.3v2.6c0 .3.3.6.7.5 3.7-1.5 6.3-5.1 6.3-9.3 0-6-5.1-10.7-11.1-10Z"
};
var spotify = {
  name: "spotify",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.489 2 2 6.489 2 12s4.489 10 10 10 10-4.489 10-10S17.511 2 12 2Zm0 2c4.43 0 8 3.57 8 8s-3.57 8-8 8-8-3.57-8-8 3.57-8 8-8Zm-1.31 4c-1.54 0-2.994.172-4.362.516-.342.085-.6.34-.6.853 0 .513.343.942.856.856.256 0 .343-.086.514-.086a16.488 16.488 0 0 1 3.592-.428c2.395 0 4.874.598 6.585 1.539.256.085.341.172.512.172.514 0 .857-.343.942-.856 0-.427-.255-.684-.512-.855C16.079 8.599 13.34 8 10.69 8Zm-.17 2.994c-1.454 0-2.48.257-3.506.514-.428.17-.6.341-.6.77 0 .341.256.683.684.683.17 0 .256 0 .427-.086.77-.171 1.797-.342 2.909-.342 2.223 0 4.276.512 5.73 1.453.171.085.343.172.514.172.427 0 .682-.342.77-.77 0-.255-.172-.512-.428-.683-1.883-1.112-4.105-1.71-6.5-1.71Zm.255 3.014c-1.197 0-2.31.17-3.421.428-.342 0-.512.255-.512.597 0 .342.255.6.597.6.086 0 .257-.086.428-.086.855-.171 1.881-.342 2.823-.342 1.71 0 3.336.427 4.619 1.197.17.085.256.17.427.17.256 0 .513-.17.684-.597 0-.342-.17-.429-.342-.6a10.864 10.864 0 0 0-5.303-1.367Z"
};
var youtube = {
  name: "youtube",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m5.68 2 1.478 5.344v2.562H8.44V7.344L9.937 2h-1.29l-.538 2.432c-.15.71-.247 1.214-.29 1.515h-.04c-.063-.42-.159-.93-.29-1.525L6.97 2H5.68Zm5.752 2.018c-.434 0-.784.084-1.051.257-.267.172-.464.448-.59.825-.125.377-.187.876-.187 1.498v.84c0 .615.054 1.107.164 1.478.11.371.295.644.556.82.261.176.62.264 1.078.264.446 0 .8-.087 1.06-.26.26-.173.45-.444.565-.818.116-.374.174-.869.174-1.485v-.84c0-.62-.059-1.118-.178-1.492-.119-.373-.308-.648-.566-.824-.258-.176-.598-.263-1.025-.263Zm2.447.113v4.314c0 .534.09.927.271 1.178.182.251.465.377.848.377.552 0 .968-.267 1.244-.8h.027l.114.706H17.4V4.131h-1.298v4.588a.635.635 0 0 1-.23.263.569.569 0 0 1-.325.104c-.132 0-.226-.054-.283-.164-.057-.11-.086-.295-.086-.553V4.131h-1.3Zm-2.477.781c.182 0 .311.095.383.287.072.191.108.495.108.91v1.8c0 .426-.036.735-.108.923-.072.188-.2.282-.38.283-.183 0-.309-.095-.378-.283-.07-.188-.103-.497-.103-.924V6.11c0-.414.035-.718.107-.91.072-.19.195-.287.371-.287ZM5 11c-1.1 0-2 .9-2 2v7c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2v-7c0-1.1-.9-2-2-2H5Zm7.049 2h1.056v2.568h.008c.095-.186.232-.335.407-.449.175-.114.364-.17.566-.17.26 0 .463.07.611.207.148.138.257.361.323.668.066.308.097.736.097 1.281v.772h.002c0 .727-.088 1.26-.264 1.602-.175.341-.447.513-.818.513-.207 0-.394-.047-.564-.142a.93.93 0 0 1-.383-.391h-.024l-.11.46h-.907V13Zm-6.563.246h3.252v.885h-1.09v5.789H6.576v-5.79h-1.09v-.884Zm11.612 1.705c.376 0 .665.07.867.207.2.138.343.354.426.645.082.292.123.695.123 1.209v.836h-1.836v.248c0 .313.008.547.027.703.02.156.057.27.115.342.058.072.148.107.27.107.164 0 .277-.064.338-.191.06-.127.094-.338.1-.635l.947.055c.005.042.007.101.007.175 0 .451-.124.788-.37 1.01-.248.223-.595.334-1.046.334-.54 0-.92-.17-1.138-.51-.218-.339-.326-.863-.326-1.574v-.851c0-.733.112-1.267.338-1.604.225-.337.612-.506 1.158-.506Zm-8.688.094h1.1v3.58c0 .217.024.373.072.465.048.093.126.139.238.139a.486.486 0 0 0 .276-.088.538.538 0 0 0 .193-.223v-3.873h1.1v4.875h-.862l-.093-.598h-.026c-.234.452-.584.678-1.05.678-.325 0-.561-.106-.715-.318-.154-.212-.233-.544-.233-.994v-3.643Zm8.664.648c-.117 0-.204.036-.26.104-.055.069-.093.182-.11.338a6.506 6.506 0 0 0-.028.71v.35h.803v-.35c0-.312-.01-.548-.032-.71-.02-.162-.059-.276-.115-.342-.056-.066-.14-.1-.258-.1Zm-3.482.036a.418.418 0 0 0-.293.127.698.698 0 0 0-.192.326v2.767a.487.487 0 0 0 .438.256.337.337 0 0 0 .277-.127c.07-.085.12-.228.149-.43.029-.2.043-.48.043-.835v-.627c0-.383-.011-.676-.035-.883-.024-.207-.067-.357-.127-.444a.3.3 0 0 0-.26-.13Z"
};
var youtube_alt = {
  name: "youtube_alt",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 4s-6.254 0-7.814.418a2.503 2.503 0 0 0-1.768 1.768C2 7.746 2 12 2 12s0 4.255.418 5.814c.23.861.908 1.538 1.768 1.768C5.746 20 12 20 12 20s6.255 0 7.814-.418a2.505 2.505 0 0 0 1.768-1.768C22 16.255 22 12 22 12s0-4.254-.418-5.814a2.505 2.505 0 0 0-1.768-1.768C18.255 4 12 4 12 4Zm0 2c2.882 0 6.49.134 7.297.35a.508.508 0 0 1 .353.353c.241.898.35 3.639.35 5.297s-.109 4.398-.35 5.297a.508.508 0 0 1-.353.353c-.805.216-4.415.35-7.297.35-2.881 0-6.49-.134-7.297-.35a.508.508 0 0 1-.353-.353C4.109 16.399 4 13.658 4 12s.109-4.399.35-5.299a.505.505 0 0 1 .353-.351C5.508 6.134 9.118 6 12 6Zm-2 2.535v6.93L16 12l-6-3.465Z"
};
var apple_app_store = {
  name: "apple_app_store",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.489 2 2 6.489 2 12s4.489 10 10 10 10-4.489 10-10S17.511 2 12 2Zm0 2c4.43 0 8 3.57 8 8s-3.57 8-8 8-8-3.57-8-8 3.57-8 8-8Zm-.611 1.64-1.748.971 1.214 2.19L9.079 12h2.287L12 10.86l1.145-2.06 1.214-2.189-1.748-.97L12 6.74l-.611-1.1Zm2.328 4.19-1.145 2.059.36.646 1.353 2.438-.015.017-.006.012h.037l1.31 2.357 1.748-.97L16.588 15H18v-2h-2.523l-1.76-3.17ZM6 13v2h1.412l-.771 1.389 1.748.97L9.699 15h3.457l-1.11-2H7c-.014 0-.025.007-.04.008L6.948 13H6Z"
};
var twitter = {
  name: "twitter",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 3c-1.103 0-2 .897-2 2v14c0 1.103.897 2 2 2h14c1.103 0 2-.897 2-2V5c0-1.103-.897-2-2-2H5Zm0 2h14l.002 14H5V5Zm9.566 2.113A2.488 2.488 0 0 0 12.08 9.6c0 .257.086.428.086.6-2.057-.086-3.857-1.114-5.057-2.571-.257.343-.343.77-.343 1.2 0 .856.429 1.544 1.115 2.144-.428-.086-.772-.173-1.115-.344 0 1.2.856 2.143 1.97 2.4-.257.086-.428.086-.685.086-.086 0-.259-.086-.43-.086.343.943 1.2 1.713 2.315 1.713-.857.6-1.972 1.03-3.086 1.03h-.6c1.114.684 2.4 1.115 3.771 1.115 4.543 0 7.03-3.773 7.03-7.03v-.343a5.786 5.786 0 0 0 1.201-1.287c-.514.258-.943.343-1.457.43.514-.343.942-.772 1.113-1.372-.429.257-.943.514-1.543.6-.429-.514-1.113-.772-1.799-.772Z"
};
var apple_logo = {
  name: "apple_logo",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16 1c-1.135 0-2.231.48-2.957 1.268-.656.717-1.055 1.99-1.055 3.013 1.172 0 2.424-.6 3.12-1.402C15.754 3.128 16 2.209 16 1ZM8.275 5.514C6.181 5.514 3 7.48 3 12.11 3 16.324 6.767 21 8.893 21h.029c1.269 0 1.712-.823 3.451-.832 1.739.009 2.183.832 3.451.832h.03c1.541 0 3.941-2.464 5.146-5.424A3.99 3.99 0 0 1 18.785 12c0-1.539.867-2.86 2.125-3.53-1.087-2.043-3.014-2.956-4.44-2.956-1.538 0-2.825 1.04-4.097 1.04-1.272 0-2.559-1.04-4.098-1.04Zm0 2c.432 0 .963.204 1.524.422.748.29 1.594.619 2.574.619.98 0 1.828-.33 2.576-.62.56-.216 1.09-.421 1.522-.421.418 0 1.068.196 1.677.683A5.977 5.977 0 0 0 16.788 12c0 1.597.62 3.08 1.69 4.18-.99 1.685-2.216 2.706-2.668 2.82-.152-.002-.261-.042-.636-.203-.577-.248-1.45-.622-2.813-.629-1.34.007-2.212.38-2.789.629-.358.154-.474.196-.64.201C7.942 18.757 5 15.438 5 12.111c0-3.473 2.207-4.596 3.275-4.597Z"
};
var instagram = {
  name: "instagram",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M8 3C5.243 3 3 5.243 3 8v8c0 2.757 2.243 5 5 5h8c2.757 0 5-2.243 5-5V8c0-2.757-2.243-5-5-5H8Zm0 2h8c1.654 0 3 1.346 3 3v8c0 1.654-1.346 3-3 3H8c-1.654 0-3-1.346-3-3V8c0-1.654 1.346-3 3-3Zm9 1a1 1 0 1 0 0 2 1 1 0 0 0 0-2Zm-5 1c-2.757 0-5 2.243-5 5s2.243 5 5 5 5-2.243 5-5-2.243-5-5-5Zm0 2c1.654 0 3 1.346 3 3s-1.346 3-3 3-3-1.346-3-3 1.346-3 3-3Z"
};
var facebook = {
  name: "facebook",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 3c-1.103 0-2 .897-2 2v14c0 1.103.897 2 2 2H19c1.103 0 2-.897 2-2V5c0-1.103-.897-2-2-2H5Zm0 2h14l.002 14h-4.588v-3.965h2.365l.352-2.725H14.43v-1.736c0-.788.22-1.32 1.35-1.32h1.427V6.822a20.013 20.013 0 0 0-2.092-.103c-2.074 0-3.494 1.266-3.494 3.59v2.006H9.277v2.724h2.344V19H5V5Z"
};
var chrome = {
  name: "chrome",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C8.728 2 5.832 3.579 4.008 6.008A9.942 9.942 0 0 0 2 12c0 5.197 3.964 9.465 9.033 9.951.318.03.64.049.967.049 5.523 0 10-4.477 10-10S17.523 2 12 2Zm0 2c2.953 0 5.532 1.613 6.918 4h-3.945C14.14 7.38 13.118 7 12 7c-1.897 0-3.526 1.07-4.373 2.627l-2.19-2.19A7.993 7.993 0 0 1 12 4ZM5.037 8.074 7 12a5 5 0 0 0 5 5c.236 0 .461-.038.69-.07l-1.018 3.054C7.414 19.81 4 16.3 4 12c0-1.427.38-2.765 1.037-3.926Zm14.238.615C19.737 9.7 20 10.82 20 12c0 4.294-3.404 7.8-7.654 7.982L15 16h-.027C16.196 15.089 17 13.643 17 12c0-.789-.2-1.525-.525-2.19l2.8-1.12ZM12 9a3 3 0 1 1 0 6 3 3 0 0 1 0-6Z"
};
var ios_logo = {
  name: "ios_logo",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7.533 2.977a4.567 4.567 0 0 0-4.556 4.556v8.889a4.567 4.567 0 0 0 4.556 4.556h8.889a4.567 4.567 0 0 0 4.556-4.556V7.533a4.567 4.567 0 0 0-4.556-4.556H7.533Zm0 2h8.889a2.522 2.522 0 0 1 2.556 2.556v8.889a2.522 2.522 0 0 1-2.556 2.556H7.533a2.522 2.522 0 0 1-2.556-2.556V7.533a2.522 2.522 0 0 1 2.556-2.556ZM6.5 9a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1Zm4 0C8.961 9 8 10.152 8 12c0 1.844.943 3 2.5 3 1.553 0 2.5-1.16 2.5-3 0-1.844-.954-3-2.5-3ZM16 9c-.558 0-1.07.18-1.435.53-.366.348-.565.853-.565 1.398 0 .53.354.96.727 1.164.372.204.774.29 1.142.39.386.105.735.2.924.303.189.104.207.107.207.287 0 .317-.097.527-.254.676-.157.15-.396.252-.746.252-.324 0-.566-.093-.727-.242-.16-.15-.273-.367-.273-.758h-1c0 .61.216 1.141.594 1.492S15.479 15 16 15c.558 0 1.07-.179 1.436-.527.366-.349.564-.855.564-1.4 0-.531-.354-.96-.727-1.165-.372-.204-.774-.29-1.142-.39-.385-.105-.735-.2-.924-.303-.189-.104-.207-.108-.207-.287 0-.316.097-.526.254-.676.157-.15.396-.252.746-.252.324 0 .566.093.727.242.16.15.273.367.273.758h1c0-.61-.216-1.141-.594-1.492S16.521 9 16 9Zm-5.5 1c1.305 0 1.5 1.253 1.5 2 0 .747-.195 2-1.5 2C9.26 14 9 12.912 9 12c0-.747.195-2 1.5-2ZM6 11v4h1v-4H6Z"
};
var linkedin = {
  name: "linkedin",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 3a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2H5Zm0 2h14v14H5V5Zm2.78 1.316c-.858 0-1.372.516-1.372 1.202 0 .686.514 1.199 1.285 1.199.857 0 1.371-.513 1.371-1.2 0-.685-.514-1.2-1.285-1.2ZM6.476 10v7H9v-7H6.477Zm4.605 0v7h2.523v-3.826c0-1.14.813-1.303 1.057-1.303s.897.245.897 1.303V17H18v-3.826C18 10.977 17.024 10 15.803 10s-1.873.407-2.198.977V10h-2.523Z"
};
var microsoft_edge = {
  name: "microsoft_edge",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20.644 8.586c-.17-.711-.441-1.448-.774-2.021-.771-1.329-1.464-2.237-3.177-3.32C14.98 2.162 13.076 2 12.17 2c-2.415 0-4.211.86-5.525 1.887C3.344 6.47 3 11 3 11s1.221-2.045 3.54-3.526C7.943 6.579 9.941 6 11.568 6 15.885 6 16 10 16 10H9c0-2 1-3 1-3s-5 2-5 7.044c0 .487-.003 1.372.248 2.283.232.843.7 1.705 1.132 2.353 1.221 1.832 3.045 2.614 3.916 2.904.996.332 2.029.416 3.01.416 2.72 0 4.877-.886 5.694-1.275v-4.172c-.758.454-2.679 1.447-5 1.447-5 0-5-4-5-4h12v-2.49s-.039-1.593-.356-2.924Z"
};
var microsoft_onedrive = {
  name: "microsoft_onedrive",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M13 5c-1.811 0-3.382.972-4.26 2.414A3.958 3.958 0 0 0 7 7a4 4 0 0 0-4 4c0 .36.063.703.152 1.035A2.493 2.493 0 0 0 3.5 17h.55c-.018-.166-.05-.329-.05-.5a4.509 4.509 0 0 1 3.287-4.334A6.01 6.01 0 0 1 13 8c1.322 0 2.57.426 3.594 1.203.42-.125.86-.186 1.304-.195A5 5 0 0 0 13 5Zm0 5a4 4 0 0 0-4 4c0 .018.006.034.006.05A2.5 2.5 0 1 0 8.5 19H21a2 2 0 1 0 0-4c-.065 0-.125.014-.19.02.117-.32.19-.66.19-1.02a3 3 0 0 0-3-3c-.68 0-1.302.235-1.805.617A3.981 3.981 0 0 0 13 10Z"
};
var google_play = {
  name: "google_play",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4.602 2.01a1.45 1.45 0 0 0-.344.002c-.019.002-.036 0-.055.004a1 1 0 0 0-.59.26A1.601 1.601 0 0 0 3 3.503v17.088c0 .432.206.984.701 1.252s1.056.146 1.424-.076l.002-.002c-.06.036.276-.16.701-.4l1.715-.973c1.4-.793 3.238-1.832 5.08-2.873 1.842-1.042 3.687-2.083 5.096-2.881l1.732-.983c.432-.245.645-.366.776-.445.388-.235.78-.69.773-1.275-.008-.586-.402-1.014-.775-1.225A21953.353 21953.353 0 0 1 7.713 3.635L6 2.662c-.426-.242-.747-.428-.715-.408h-.002a1.764 1.764 0 0 0-.681-.244ZM5 5.35l5.756 6.607L5 18.567V5.35Zm3.596 1.084 3.177 1.797 1.493.843-1.184 1.36-3.486-4Zm6.445 3.642 3.338 1.887c-.426.242-.976.556-1.647.936l-1.683.953a.999.999 0 0 0-.063-.084l-1.578-1.81 1.578-1.813a.999.999 0 0 0 .055-.069Zm-2.959 3.404 1.195 1.372-1.638.927-3.073 1.739 3.516-4.038Z"
};
var google_maps = {
  name: "google_maps",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18.5 2A3.5 3.5 0 0 0 15 5.5c0 2.625 3.063 3.927 3.063 7 0 .241.196.453.437.453s.473-.178.473-.418C18.972 9.461 22 8 22 5.5A3.5 3.5 0 0 0 18.5 2ZM5 3a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7s-.125 2.375-2 3v2.586L13.414 12l1.73-1.73-1.226-1.602L5 17.586V5h8c.125-1.25.625-2 .625-2H5Zm13.533 1.299a1.167 1.167 0 1 1 .002 2.334 1.167 1.167 0 0 1-.002-2.334ZM8.502 6a2.5 2.5 0 1 0 0 5c2.099 0 2.56-1.963 2.355-2.938l-2.355-.001v.955h1.36c-.179.579-.66.992-1.36.992A1.51 1.51 0 0 1 6.992 8.5 1.51 1.51 0 0 1 9.48 7.355l.705-.703A2.493 2.493 0 0 0 8.502 6ZM12 13.414 17.586 19H6.414L12 13.414Z"
};
var microsoft_outlook = {
  name: "microsoft_outlook",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14 2 2 4v16l12 2V2Zm1 5v2.25l2 1.156 4-2.312v1.594L17 12l-2-1.156V15h1.344c.562-1.18 1.761-2 3.156-2 .977 0 1.863.387 2.5 1.031V8c0-.602-.398-1-1-1h-6ZM7.594 8c1.02 0 1.832.371 2.468 1.094.63.722.938 1.664.938 2.844 0 1.21-.309 2.226-.969 2.968C9.383 15.648 8.52 16 7.47 16c-1.028 0-1.864-.371-2.5-1.094C4.32 14.184 4 13.254 4 12.094c0-1.223.309-2.215.969-2.969C5.629 8.371 6.512 8 7.594 8ZM7.53 9.5a1.57 1.57 0 0 0-1.343.688c-.329.457-.5 1.058-.5 1.812 0 .773.171 1.395.5 1.844.328.449.765.656 1.312.656.555 0 .992-.23 1.313-.656.328-.438.5-1.04.5-1.813 0-.793-.16-1.406-.47-1.844A1.544 1.544 0 0 0 7.532 9.5ZM19.5 14a2.497 2.497 0 0 0-2.5 2.5c0 1.383 1.117 2.5 2.5 2.5s2.5-1.117 2.5-2.5-1.117-2.5-2.5-2.5Zm-.5 1h1v1h1v1h-2v-2Z"
};
var power_bi = {
  name: "power_bi",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h2v-2H4V6h16v12h-1v2h1c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2ZM7 15h2v5H7v-5ZM10 13h2v7h-2v-7ZM13 14h2v6h-2v-6ZM16 11h2v9h-2v-9Z"
};
var slack = {
  name: "slack",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7.362 14.11a1.682 1.682 0 0 1-3.362 0 1.688 1.688 0 0 1 1.681-1.682h1.681v1.681Zm.847 0a1.688 1.688 0 0 1 1.682-1.682 1.688 1.688 0 0 1 1.68 1.681v4.21A1.688 1.688 0 0 1 9.892 20a1.684 1.684 0 0 1-1.682-1.681v-4.21Zm1.682-6.747A1.684 1.684 0 0 1 8.209 5.68 1.688 1.688 0 0 1 9.891 4a1.688 1.688 0 0 1 1.68 1.681v1.681h-1.68Zm0 .846a1.686 1.686 0 0 1 1.68 1.682 1.686 1.686 0 0 1-1.68 1.68H5.68A1.688 1.688 0 0 1 4 9.892a1.684 1.684 0 0 1 1.681-1.682h4.21Zm6.746 1.682a1.682 1.682 0 0 1 1.682-1.682A1.688 1.688 0 0 1 20 9.891a1.688 1.688 0 0 1-1.681 1.68h-1.681v-1.68Zm-.846 0a1.688 1.688 0 0 1-1.682 1.68 1.686 1.686 0 0 1-1.68-1.68V5.68A1.688 1.688 0 0 1 14.108 4a1.682 1.682 0 0 1 1.682 1.681v4.21Zm-1.682 6.746a1.68 1.68 0 0 1 1.682 1.682A1.686 1.686 0 0 1 14.109 20a1.688 1.688 0 0 1-1.68-1.681v-1.681h1.68Zm0-.846a1.688 1.688 0 0 1-1.68-1.682 1.686 1.686 0 0 1 1.68-1.68h4.21A1.688 1.688 0 0 1 20 14.108a1.682 1.682 0 0 1-1.681 1.682h-4.21Z"
};
var blocked_off = {
  name: "blocked_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20.5 11.5c0-4.41-3.59-8-8-8-1.41 0-2.73.37-3.87 1.01L7.17 3.05A9.9 9.9 0 0 1 12.5 1.5c5.52 0 10 4.48 10 10 0 1.96-.57 3.79-1.55 5.34l-1.46-1.46a7.95 7.95 0 0 0 1.01-3.88Zm-5.88-1h2.88v2h-.88l-2-2ZM2.91 1.63 1.5 3.04l2.78 2.78A9.92 9.92 0 0 0 2.5 11.5c0 5.52 4.48 10 10 10 2.11 0 4.07-.66 5.68-1.78l2.78 2.78 1.41-1.41L2.91 1.63ZM4.5 11.5c0 4.41 3.59 8 8 8 1.56 0 3-.45 4.23-1.23l-5.77-5.77H7.5v-2h1.46L5.73 7.27A7.846 7.846 0 0 0 4.5 11.5Z"
};
var blocked = {
  name: "blocked",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2Zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8Zm5-9H7v2h10v-2Z"
};
var security = {
  name: "security",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m3 5 9-4 9 4v6c0 5.55-3.84 10.74-9 12-5.16-1.26-9-6.45-9-12V5Zm16 6.99h-7v-8.8L5 6.3V12h7v8.93c3.72-1.15 6.47-4.82 7-8.94Z"
};
var flagged_off = {
  name: "flagged_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4.5 3.5h9l.4 2h5.6v10h-7l-.4-2H6.5v7h-2v-17Zm7.44 2.39-.08-.39H6.5v6h7.24l.32 1.61.08.39h3.36v-6h-5.24l-.32-1.61Z"
};
var lock_add = {
  name: "lock_add",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17 8.5h1c1.1 0 2 .9 2 2v10c0 1.1-.9 2-2 2H6c-1.1 0-2-.9-2-2v-10c0-1.1.9-2 2-2h1v-2c0-2.76 2.24-5 5-5s5 2.24 5 5v2Zm-5-5.1c-1.71 0-3.1 1.39-3.1 3.1v2h6.2v-2c0-1.71-1.39-3.1-3.1-3.1ZM6 20.5v-10h12v10H6Zm5-6v-3h2v3h3v2h-3v3h-2v-3H8v-2h3Z"
};
var lock_off = {
  name: "lock_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M8.805 6c0-1.71 1.39-3.1 3.1-3.1 1.71 0 3.1 1.39 3.1 3.1v2h-4.66l2 2h5.56v5.56l2 2V10c0-1.1-.9-2-2-2h-1V6c0-2.76-2.24-5-5-5-2.32 0-4.26 1.59-4.82 3.74l1.72 1.72V6Zm-4.49-1.19-1.41 1.41 2.04 2.04c-.62.34-1.04.99-1.04 1.74v10c0 1.1.9 2 2 2h12.78l1 1 1.41-1.41L4.315 4.81ZM5.905 10v10h10.78l-10-10h-.78Z"
};
var lock = {
  name: "lock",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17 8.5h1c1.1 0 2 .9 2 2v10c0 1.1-.9 2-2 2H6c-1.1 0-2-.9-2-2v-10c0-1.1.9-2 2-2h1v-2c0-2.76 2.24-5 5-5s5 2.24 5 5v2Zm-5-5c-1.66 0-3 1.34-3 3v2h6v-2c0-1.66-1.34-3-3-3Zm-6 17v-10h12v10H6Zm8-5c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2Z"
};
var lock_open = {
  name: "lock_open",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18 8.5h-1v-2c0-2.76-2.24-5-5-5s-5 2.24-5 5h2c0-1.66 1.34-3 3-3s3 1.34 3 3v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2v-10c0-1.1-.9-2-2-2Zm-12 12v-10h12v10H6Zm8-5c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2Z"
};
var verified = {
  name: "verified",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 1H5c-1.1 0-1.99.9-1.99 2L3 15.93c0 .69.35 1.3.88 1.66L12 23l8.11-5.41c.53-.36.88-.97.88-1.66L21 3c0-1.1-.9-2-2-2Zm-7 19.6-7-4.66V3h14v12.93l-7 4.67ZM7.41 10.59l2.58 2.59 6.59-6.6L18 8l-8 8-4-4 1.41-1.41Z"
};
var verified_user = {
  name: "verified_user",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 1 3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4Zm7 10c0 4.52-2.98 8.69-7 9.93-4.02-1.24-7-5.41-7-9.93V6.3l7-3.11 7 3.11V11ZM6 13l1.41-1.41L10 14.17l6.59-6.59L18 9l-8 8-4-4Z"
};
var flagged = {
  name: "flagged",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14.4 6 14 4H5v17h2v-7h5.6l.4 2h7V6h-5.6Z"
};
var visibility = {
  name: "visibility",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 6a9.77 9.77 0 0 1 8.82 5.5A9.77 9.77 0 0 1 12 17a9.77 9.77 0 0 1-8.82-5.5A9.77 9.77 0 0 1 12 6Zm0-2C7 4 2.73 7.11 1 11.5 2.73 15.89 7 19 12 19s9.27-3.11 11-7.5C21.27 7.11 17 4 12 4Zm0 5a2.5 2.5 0 0 1 0 5 2.5 2.5 0 0 1 0-5Zm0-2c-2.48 0-4.5 2.02-4.5 4.5S9.52 16 12 16s4.5-2.02 4.5-4.5S14.48 7 12 7Z"
};
var key = {
  name: "key",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M22 19h-6v-4h-2.68c-1.14 2.42-3.6 4-6.32 4-3.86 0-7-3.14-7-7s3.14-7 7-7c2.72 0 5.17 1.58 6.32 4H24v6h-2v4Zm-4-2h2v-4h2v-2H11.94l-.23-.67C11.01 8.34 9.11 7 7 7c-2.76 0-5 2.24-5 5s2.24 5 5 5c2.11 0 4.01-1.34 4.71-3.33l.23-.67H18v4ZM7 15c-1.65 0-3-1.35-3-3s1.35-3 3-3 3 1.35 3 3-1.35 3-3 3Zm0-4c-.55 0-1 .45-1 1s.45 1 1 1 1-.45 1-1-.45-1-1-1Z"
};
var visibility_off = {
  name: "visibility_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 6a9.77 9.77 0 0 1 8.82 5.5 9.647 9.647 0 0 1-2.41 3.12l1.41 1.41c1.39-1.23 2.49-2.77 3.18-4.53C21.27 7.11 17 4 12 4c-1.27 0-2.49.2-3.64.57l1.65 1.65C10.66 6.09 11.32 6 12 6Zm-1.07 1.14L13 9.21c.57.25 1.03.71 1.28 1.28l2.07 2.07c.08-.34.14-.7.14-1.07C16.5 9.01 14.48 7 12 7c-.37 0-.72.05-1.07.14ZM2.01 3.87l2.68 2.68A11.738 11.738 0 0 0 1 11.5C2.73 15.89 7 19 12 19c1.52 0 2.98-.29 4.32-.82l3.42 3.42 1.41-1.41L3.42 2.45 2.01 3.87Zm7.5 7.5 2.61 2.61c-.04.01-.08.02-.12.02a2.5 2.5 0 0 1-2.5-2.5c0-.05.01-.08.01-.13Zm-3.4-3.4 1.75 1.75a4.6 4.6 0 0 0-.36 1.78 4.507 4.507 0 0 0 6.27 4.14l.98.98c-.88.24-1.8.38-2.75.38a9.77 9.77 0 0 1-8.82-5.5c.7-1.43 1.72-2.61 2.93-3.53Z"
};
var business = {
  name: "business",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 7h10v14H2V3h10v4ZM4 19h2v-2H4v2Zm2-4H4v-2h2v2Zm-2-4h2V9H4v2Zm2-4H4V5h2v2Zm2 12h2v-2H8v2Zm2-4H8v-2h2v2Zm-2-4h2V9H8v2Zm2-4H8V5h2v2Zm10 12V9h-8v2h2v2h-2v2h2v2h-2v2h8Zm-2-8h-2v2h2v-2Zm-2 4h2v2h-2v-2Z"
};
var meeting_room = {
  name: "meeting_room",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 19V4h-4V3H5v16H3v2h12V6h2v15h4v-2h-2Zm-6 0H7V5h6v14Zm-1-8h-2v2h2v-2Z"
};
var meeting_room_off = {
  name: "meeting_room_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m2.975 1.565-1.41 1.41 4 4v11.46h-2v2h11v-4.46l6.46 6.46 1.41-1.41-19.46-19.46Zm9.59 2.87v3.88l2 2v-4.88h3v7.88l2 2V3.435h-5v-1h-7.88l2 2h3.88Zm-5 14h5v-4.46l-5-5v9.46Z"
};
var pool = {
  name: "pool",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m10 8-3.25 3.25c.28.108.51.241.707.354l.063.036c.37.23.59.36 1.15.36.56 0 .78-.13 1.15-.36l.013-.007c.458-.27 1.077-.633 2.177-.633 1.11 0 1.73.37 2.18.64l.01.007c.365.216.595.353 1.14.353.55 0 .78-.13 1.15-.36.12-.07.26-.15.41-.23L10.48 5C8.93 3.45 7.5 2.99 5 3v2.5c1.82-.01 2.89.39 4 1.5l1 1Zm-3.51 7.854c-.365-.217-.595-.354-1.14-.354-.55 0-.78.13-1.15.36l-.03.017c-.466.268-1.083.623-2.17.623v-2c.56 0 .78-.13 1.15-.36.45-.27 1.07-.64 2.18-.64s1.73.37 2.18.64l.01.007c.365.216.595.353 1.14.353.56 0 .78-.13 1.15-.36.45-.27 1.07-.64 2.18-.64s1.73.37 2.18.64l.01.007c.364.216.595.353 1.14.353.55 0 .78-.13 1.15-.36.45-.27 1.07-.64 2.18-.64s1.73.37 2.18.64l.01.007c.364.216.595.353 1.14.353v2c-1.1-.01-1.71-.37-2.16-.64l-.01-.006c-.364-.217-.595-.354-1.14-.354-.56 0-.78.13-1.15.36-.45.27-1.07.64-2.18.64s-1.73-.37-2.18-.64l-.01-.006c-.364-.217-.595-.354-1.14-.354-.56 0-.78.13-1.15.36-.45.27-1.07.64-2.18.64s-1.73-.37-2.18-.64l-.01-.006ZM18.67 18c-1.11 0-1.73.37-2.18.64-.37.23-.6.36-1.15.36-.545 0-.775-.137-1.14-.353l-.01-.007c-.45-.27-1.07-.64-2.18-.64-1.1 0-1.719.363-2.177.633l-.013.007c-.37.23-.59.36-1.15.36-.56 0-.78-.13-1.15-.36-.45-.27-1.07-.64-2.18-.64-1.1 0-1.719.363-2.177.633l-.013.007c-.37.23-.59.36-1.15.36v2c1.1 0 1.719-.363 2.177-.632l.013-.008c.37-.23.6-.36 1.15-.36.55 0 .78.13 1.15.36.45.27 1.07.64 2.18.64 1.1 0 1.719-.363 2.177-.632l.013-.008c.37-.23.59-.36 1.15-.36.545 0 .776.137 1.14.354l.01.006c.45.27 1.07.64 2.18.64 1.09 0 1.698-.357 2.156-.625l.024-.015c.37-.23.59-.36 1.15-.36.545 0 .775.137 1.14.354l.01.006c.45.27 1.07.64 2.18.64v-2c-.56 0-.78-.13-1.15-.36-.45-.27-1.07-.64-2.18-.64ZM14 5.5a2.5 2.5 0 1 1 5 0 2.5 2.5 0 0 1-5 0Z"
};
var cafe = {
  name: "cafe",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 3h16c1.11 0 2 .89 2 2v3a2 2 0 0 1-2 2h-2v3c0 2.21-1.79 4-4 4H7c-2.21 0-4-1.79-4-4V3Zm16 16H3v2h16v-2Zm-4-6c0 1.1-.9 2-2 2H7c-1.1 0-2-.9-2-2V5h10v8Zm2-5h2V5h-2v3Z"
};
var gym = {
  name: "gym",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20.57 14.86 22 13.43 20.57 12 17 15.57 8.43 7 12 3.43 10.57 2 9.14 3.43 7.71 2 5.57 4.14 4.14 2.71 2.71 4.14l1.43 1.43L2 7.71l1.43 1.43L2 10.57 3.43 12 7 8.43 15.57 17 12 20.57 13.43 22l1.43-1.43L16.29 22l2.14-2.14 1.43 1.43 1.43-1.43-1.43-1.43L22 16.29l-1.43-1.43Z"
};
var beach = {
  name: "beach",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M13.115 3.001c-2.58 0-5.16.98-7.14 2.95l-.01.01c-3.95 3.95-3.95 10.36 0 14.31l14.3-14.31a10.086 10.086 0 0 0-7.15-2.96Zm7.882 16.57-1.429 1.428-6.441-6.442 1.428-1.428 6.442 6.441ZM4.995 13.12c0 1.49.4 2.91 1.14 4.15l1.39-1.38a11.285 11.285 0 0 1-2.07-5.44c-.3.85-.46 1.74-.46 2.67Zm3.98 1.31c-1.35-2.05-1.86-4.5-1.38-6.83.58-.12 1.16-.18 1.75-.18 1.8 0 3.55.55 5.08 1.56l-5.45 5.45Zm4.14-9.43c-.93 0-1.82.16-2.67.46 1.91.19 3.78.89 5.43 2.07l1.39-1.39a8.063 8.063 0 0 0-4.15-1.14Z"
};
var world = {
  name: "world",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2ZM4 12c0-.61.08-1.21.21-1.78L8.99 15v1c0 1.1.9 2 2 2v1.93C7.06 19.43 4 16.07 4 12Zm11.99 4c.9 0 1.64.59 1.9 1.4A7.991 7.991 0 0 0 20 12c0-3.35-2.08-6.23-5.01-7.41V5c0 1.1-.9 2-2 2h-2v2c0 .55-.45 1-1 1h-2v2h6c.55 0 1 .45 1 1v3h1Z"
};
var school = {
  name: "school",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 3 1 9l4 2.18v6L12 21l7-3.82v-6l2-1.09V17h2V9L12 3Zm6.82 6L12 12.72 5.18 9 12 5.28 18.82 9ZM12 18.72l5-2.73v-3.72L12 15l-5-2.73v3.72l5 2.73Z"
};
var city = {
  name: "city",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15 5.5v6h6v10H3v-14h6v-2l3-3 3 3Zm-10 14h2v-2H5v2Zm2-4H5v-2h2v2Zm-2-4h2v-2H5v2Zm6 8v-2h2v2h-2Zm0-6v2h2v-2h-2Zm0-2v-2h2v2h-2Zm0-6v2h2v-2h-2Zm8 14h-2v-2h2v2Zm-2-4h2v-2h-2v2Z"
};
var account_circle = {
  name: "account_circle",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2ZM7.07 18.28c.43-.9 3.05-1.78 4.93-1.78s4.51.88 4.93 1.78A7.893 7.893 0 0 1 12 20c-1.86 0-3.57-.64-4.93-1.72ZM12 14.5c1.46 0 4.93.59 6.36 2.33A7.95 7.95 0 0 0 20 12c0-4.41-3.59-8-8-8s-8 3.59-8 8c0 1.82.62 3.49 1.64 4.83 1.43-1.74 4.9-2.33 6.36-2.33ZM12 6c-1.94 0-3.5 1.56-3.5 3.5S10.06 13 12 13s3.5-1.56 3.5-3.5S13.94 6 12 6Zm-1.5 3.5c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5S12.83 8 12 8s-1.5.67-1.5 1.5Z"
};
var users_circle = {
  name: "users_circle",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2Zm.51 7.99c0-1.65-1.35-3-3-3s-3 1.35-3 3 1.35 3 3 3 3-1.35 3-3Zm-3 1c-.55 0-1-.45-1-1s.45-1 1-1 1 .45 1 1-.45 1-1 1Zm8.5 0c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2-.01-1.11.89-2 2-2 1.11 0 2 .89 2 2ZM9.51 16c-1.39 0-2.98.57-3.66 1.11a7.935 7.935 0 0 0 5.66 2.86v-2.78c0-1.89 2.98-2.7 4.5-2.7.88 0 2.24.28 3.24.87.48-1.03.75-2.17.75-3.37 0-4.41-3.59-8-8-8s-8 3.59-8 8c0 1.23.28 2.39.78 3.43 1.34-.98 3.43-1.43 4.73-1.43.44 0 .97.06 1.53.16-.63.57-1.06 1.22-1.3 1.86-.041 0-.083-.003-.123-.005A1.646 1.646 0 0 0 9.51 16Z"
};
var face = {
  name: "face",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M22 12c0 5.52-4.48 10-10 10S2 17.52 2 12 6.48 2 12 2s10 4.48 10 10ZM9 14.25a1.25 1.25 0 1 0 0-2.5 1.25 1.25 0 0 0 0 2.5ZM13.75 13a1.25 1.25 0 1 1 2.5 0 1.25 1.25 0 0 1-2.5 0Zm3.75-5c-2.9 0-5.44-1.56-6.84-3.88.43-.07.88-.12 1.34-.12 2.9 0 5.44 1.56 6.84 3.88-.43.07-.88.12-1.34.12ZM4.42 9.47a8.046 8.046 0 0 0 3.66-4.44 8.046 8.046 0 0 0-3.66 4.44Zm15.25.29c.21.71.33 1.46.33 2.24 0 4.41-3.59 8-8 8s-8-3.59-8-8l.002-.05c.002-.033.005-.064-.002-.09 2.6-.98 4.69-2.99 5.74-5.55A10 10 0 0 0 17.5 10c.75 0 1.47-.09 2.17-.24Z"
};
var group = {
  name: "group",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M9 12c1.93 0 3.5-1.57 3.5-3.5S10.93 5 9 5 5.5 6.57 5.5 8.5 7.07 12 9 12Zm-7 5.25c0-2.33 4.66-3.5 7-3.5s7 1.17 7 3.5V19H2v-1.75Zm7-1.5c-1.79 0-3.82.67-4.66 1.25h9.32c-.84-.58-2.87-1.25-4.66-1.25Zm1.5-7.25C10.5 7.67 9.83 7 9 7s-1.5.67-1.5 1.5S8.17 10 9 10s1.5-.67 1.5-1.5Zm5.54 5.31c1.16.84 1.96 1.96 1.96 3.44V19h4v-1.75c0-2.02-3.5-3.17-5.96-3.44ZM18.5 8.5c0 1.93-1.57 3.5-3.5 3.5-.54 0-1.04-.13-1.5-.35.63-.89 1-1.98 1-3.15s-.37-2.26-1-3.15c.46-.22.96-.35 1.5-.35 1.93 0 3.5 1.57 3.5 3.5Z"
};
var group_add = {
  name: "group_add",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 12c1.93 0 3.5-1.57 3.5-3.5S13.93 5 12 5 8.5 6.57 8.5 8.5 10.07 12 12 12Zm-7 3v-3h3v-2H5V7H3v3H0v2h3v3h2Zm7-1.25c-2.34 0-7 1.17-7 3.5V19h14v-1.75c0-2.33-4.66-3.5-7-3.5Zm0 2c-1.79 0-3.82.67-4.66 1.25h9.32c-.84-.58-2.87-1.25-4.66-1.25Zm1.5-7.25c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5.67 1.5 1.5 1.5 1.5-.67 1.5-1.5ZM17 12c1.93 0 3.5-1.57 3.5-3.5S18.93 5 17 5c-.24 0-.48.02-.71.07a5.416 5.416 0 0 1-.02 6.85c.24.05.48.08.73.08Zm4 5.25c0-1.36-.68-2.42-1.68-3.23 2.24.47 4.68 1.54 4.68 3.23V19h-3v-1.75Z"
};
var person = {
  name: "person",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 4C9.79 4 8 5.79 8 8s1.79 4 4 4 4-1.79 4-4-1.79-4-4-4Zm2 4c0-1.1-.9-2-2-2s-2 .9-2 2 .9 2 2 2 2-.9 2-2Zm4 10c-.2-.71-3.3-2-6-2-2.69 0-5.77 1.28-6 2h12ZM4 18c0-2.66 5.33-4 8-4s8 1.34 8 4v2H4v-2Z"
};
var person_add = {
  name: "person_add",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4Zm0-6c1.1 0 2 .9 2 2s-.9 2-2 2-2-.9-2-2 .9-2 2-2ZM7 18c0-2.66 5.33-4 8-4s8 1.34 8 4v2H7v-2Zm2 0c.22-.72 3.31-2 6-2 2.7 0 5.8 1.29 6 2H9Zm-3-6v3H4v-3H1v-2h3V7h2v3h3v2H6Z"
};
var baby = {
  name: "baby",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20.13 8.17c1.45.43 2.56 1.66 2.81 3.17.04.21.06.43.06.66 0 .23-.02.45-.06.66a3.998 3.998 0 0 1-2.8 3.17 9.086 9.086 0 0 1-2.17 2.89A8.93 8.93 0 0 1 12 21c-2.29 0-4.38-.86-5.96-2.28-.9-.8-1.65-1.78-2.17-2.89a4.008 4.008 0 0 1-2.81-3.17C1.02 12.45 1 12.23 1 12c0-.23.02-.45.06-.66a3.994 3.994 0 0 1 2.81-3.17c.52-1.11 1.27-2.1 2.19-2.91A8.885 8.885 0 0 1 12 3c2.28 0 4.36.85 5.94 2.26.91.81 1.66 1.79 2.19 2.91ZM9.5 9.25a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5Zm5 0a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5ZM19 14a1.34 1.34 0 0 1-.189-.017c-.033-.005-.066-.01-.101-.013-.2.67-.49 1.29-.86 1.86A6.976 6.976 0 0 1 12 19c-2.45 0-4.6-1.26-5.85-3.17-.37-.57-.66-1.19-.86-1.86-.035.003-.068.008-.101.013A1.339 1.339 0 0 1 5 14c-1.1 0-2-.9-2-2s.9-2 2-2c.065 0 .126.008.189.017.033.005.066.01.101.013.2-.67.49-1.29.86-1.86A6.976 6.976 0 0 1 12 5c2.45 0 4.6 1.26 5.85 3.17.37.57.66 1.19.86 1.86.035-.003.068-.008.101-.013A1.34 1.34 0 0 1 19 10c1.1 0 2 .9 2 2s-.9 2-2 2Zm-2.5 0c-.76 1.77-2.49 3-4.5 3s-3.74-1.23-4.5-3h9Z"
};
var badge = {
  name: "badge",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11 2a2 2 0 0 0-2 2v3H4a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-5V4a2 2 0 0 0-2-2h-2Zm3.732 7A2 2 0 0 1 13 10h-2a2 2 0 0 1-1.732-1H4v11h16V9h-5.268ZM11 4h2v4h-2V4ZM8.5 11a1.75 1.75 0 1 0-.001 3.499A1.75 1.75 0 0 0 8.5 11Zm.875 1.75a.878.878 0 0 0-.875-.875.878.878 0 0 0-.875.875c0 .481.394.875.875.875a.878.878 0 0 0 .875-.875Zm1.75 4.375c-.088-.31-1.444-.875-2.625-.875-1.177 0-2.524.56-2.625.875h5.25Zm-6.125 0c0-1.164 2.332-1.75 3.5-1.75 1.168 0 3.5.586 3.5 1.75V18H5v-.875ZM14 11h4v1h-4zM14 14h4v1h-4zM14 17h4v1h-4z"
};
var shopping_cart_add = {
  name: "shopping_cart_add",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M13.92 9.5h-2v-3h-3v-2h3v-3h2v3h3v2h-3v3Zm-7.99 11c0-1.1.89-2 1.99-2s2 .9 2 2-.9 2-2 2-1.99-.9-1.99-2Zm11.99-2c-1.1 0-1.99.9-1.99 2s.89 2 1.99 2 2-.9 2-2-.9-2-2-2Zm-1.45-5H9.02l-1.1 2h12v2h-12c-1.52 0-2.48-1.63-1.75-2.97l1.35-2.44-3.6-7.59h-2v-2h3.27l4.26 9h7.02l3.87-7 1.74.96-3.86 7.01c-.34.62-1 1.03-1.75 1.03Z"
};
var shopping_basket = {
  name: "shopping_basket",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17.21 9.49H22c.55 0 1 .45 1 1l-.03.27-2.54 9.27a2.01 2.01 0 0 1-1.93 1.46h-13c-.92 0-1.69-.62-1.92-1.46l-2.54-9.27a.842.842 0 0 1-.04-.27c0-.55.45-1 1-1h4.79l4.38-6.55c.19-.29.51-.43.83-.43.32 0 .64.14.83.42l4.38 6.56Zm-2.41 0L12 5.29l-2.8 4.2h5.6Zm3.7 10-12.99.01-2.2-8.01H20.7l-2.2 8Zm-8.5-4c0-1.1.9-2 2-2s2 .9 2 2-.9 2-2 2-2-.9-2-2Z"
};
var shopping_card = {
  name: "shopping_card",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18.296 11.97c-.34.62-1 1.03-1.75 1.03h-7.45l-1.1 2h12v2h-12c-1.52 0-2.48-1.63-1.75-2.97l1.35-2.44L3.996 4h-2V2h3.27l.94 2h14.8c.76 0 1.24.82.87 1.48l-3.58 6.49ZM19.306 6H7.156l2.37 5h7.02l2.76-5ZM7.996 18c-1.1 0-1.99.9-1.99 2s.89 2 1.99 2 2-.9 2-2-.9-2-2-2Zm8.01 2c0-1.1.89-2 1.99-2s2 .9 2 2-.9 2-2 2-1.99-.9-1.99-2Z"
};
var shopping_cart_off = {
  name: "shopping_cart_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m.565 1.975 1.41-1.41 21.46 21.46-1.41 1.41-2.84-2.84c-.36.51-.95.84-1.62.84a1.997 1.997 0 0 1-1.16-3.62l-1.38-1.38h-7.46c-1.1 0-2-.9-2-2 0-.35.09-.68.25-.96l1.35-2.45-2.21-4.66-4.39-4.39Zm8.1 10.46-1.1 2h5.46l-2-2h-2.36Zm11.9-9H7.685l2 2h9.19l-2.76 5h-1.44l1.94 1.94c.54-.14.99-.49 1.25-.97l3.58-6.49c.37-.66-.12-1.48-.88-1.48Zm-14.99 16c0-1.1.89-2 1.99-2s2 .9 2 2-.9 2-2 2-1.99-.9-1.99-2Z"
};
var credit_card = {
  name: "credit_card",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20 4H4c-1.11 0-1.99.89-1.99 2L2 18c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V6c0-1.11-.89-2-2-2Zm0 14H4v-6h16v6ZM4 8h16V6H4v2Z"
};
var receipt = {
  name: "receipt",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19.5 3.5 18 2l-1.5 1.5L15 2l-1.5 1.5L12 2l-1.5 1.5L9 2 7.5 3.5 6 2 4.5 3.5 3 2v20l1.5-1.5L6 22l1.5-1.5L9 22l1.5-1.5L12 22l1.5-1.5L15 22l1.5-1.5L18 22l1.5-1.5L21 22V2l-1.5 1.5ZM5 19.09V4.91h14v14.18H5ZM18 17v-2H6v2h12Zm0-6v2H6v-2h12Zm0-2V7H6v2h12Z"
};
var notifications_important = {
  name: "notifications_important",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M10.5 2.75c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5v1.17c3.14.68 5.5 3.48 5.5 6.83v6l2 2v1H3v-1l2-2v-6C5 7.4 7.36 4.6 10.5 3.92V2.75Zm1.5 3c2.76 0 5 2.24 5 5v7H7v-7c0-2.76 2.24-5 5-5Zm-1.99 15.01c0 1.1.89 1.99 1.99 1.99s1.99-.89 1.99-1.99h-3.98ZM13 7.75v4h-2v-4h2Zm0 8v-2h-2v2h2Z"
};
var notifications_add = {
  name: "notifications_add",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 1.25c-.83 0-1.5.67-1.5 1.5v1.17C7.36 4.6 5 7.4 5 10.75v6l-2 2v1h18v-1l-2-2v-6c0-3.35-2.36-6.15-5.5-6.83V2.75c0-.83-.67-1.5-1.5-1.5Zm5 9.5c0-2.76-2.24-5-5-5s-5 2.24-5 5v7h10v-7Zm-5 12c-1.1 0-1.99-.89-1.99-1.99h3.98c0 1.1-.89 1.99-1.99 1.99Zm-1-15h2v3h3v2h-3v3h-2v-3H8v-2h3v-3Z"
};
var do_not_disturb = {
  name: "do_not_disturb",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2Zm0 18c-4.42 0-8-3.58-8-8 0-1.85.63-3.55 1.69-4.9L16.9 18.31A7.902 7.902 0 0 1 12 20ZM7.1 5.69 18.31 16.9A7.902 7.902 0 0 0 20 12c0-4.42-3.58-8-8-8-1.85 0-3.55.63-4.9 1.69Z"
};
var notifications_paused = {
  name: "notifications_paused",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18 10.75v5l2 2v1H4v-1l2-2v-5c0-3.08 1.64-5.64 4.5-6.32v-.68c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5v.68c2.87.68 4.5 3.25 4.5 6.32Zm-5.7-1.2H9.5v-1.8h5v1.8l-2.8 3.4h2.8v1.8h-5v-1.8l2.8-3.4Zm3.7 7.2H8v-6c0-2.48 1.51-4.5 4-4.5s4 2.02 4 4.5v6Zm-2 3c0 1.1-.9 2-2 2s-2-.9-2-2h4Z"
};
var warning_outlined = {
  name: "warning_outlined",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m1 21.5 11-19 11 19H1Zm18.53-2L12 6.49 4.47 19.5h15.06Zm-8.53-3v2h2v-2h-2Zm0-6h2v4h-2v-4Z"
};
var error_outlined = {
  name: "error_outlined",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2ZM13 13V7h-2v6h2Zm0 4v-2h-2v2h2Zm-9-5c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8-8 3.58-8 8Z"
};
var sync = {
  name: "sync",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 1v3c4.42 0 8 3.58 8 8 0 1.57-.46 3.03-1.24 4.26L17.3 14.8c.45-.83.7-1.79.7-2.8 0-3.31-2.69-6-6-6v3L8 5l4-4ZM6 12c0 3.31 2.69 6 6 6v-3l4 4-4 4v-3c-4.42 0-8-3.58-8-8 0-1.57.46-3.03 1.24-4.26L6.7 9.2c-.45.83-.7 1.79-.7 2.8Z"
};
var sync_off = {
  name: "sync_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m3.32 4.73 1.41-1.41 15.95 15.95-1.41 1.41-2.58-2.58c-.68.42-1.43.75-2.23.96v-2.09c.26-.1.51-.21.76-.34L7.14 8.55c-.43.83-.68 1.77-.68 2.77 0 1.66.68 3.15 1.76 4.24l2.24-2.24v6h-6l2.36-2.36a7.925 7.925 0 0 1-1.14-9.87L3.32 4.73Zm17.14 6.59c0-2.21-.91-4.2-2.36-5.64l2.36-2.36h-6v6l2.24-2.24a6.003 6.003 0 0 1 1.76 4.24c0 .85-.19 1.65-.51 2.38l1.5 1.5a7.921 7.921 0 0 0 1.01-3.88Zm-10-5.65V3.58c-.66.17-1.29.43-1.88.75l1.5 1.5c.065-.025.128-.053.19-.08.063-.028.125-.055.19-.08Z"
};
var sync_problem = {
  name: "sync_problem",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21 4h-6v6l2.24-2.24A6.003 6.003 0 0 1 19 12a5.99 5.99 0 0 1-4 5.65v2.09c3.45-.89 6-4.01 6-7.74 0-2.21-.91-4.2-2.36-5.64L21 4ZM5.36 17.64A7.925 7.925 0 0 1 3 12c0-3.73 2.55-6.85 6-7.74v2.09C6.67 7.17 5 9.39 5 12c0 1.66.68 3.15 1.76 4.24L9 14v6H3l2.36-2.36ZM13 17h-2v-2h2v2Zm0-4h-2V7h2v6Z"
};
var notifications = {
  name: "notifications",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18 15.75v-5c0-3.07-1.63-5.64-4.5-6.32v-.68c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v.68C7.64 5.11 6 7.67 6 10.75v5l-2 2v1h16v-1l-2-2Zm-6 6c1.1 0 2-.9 2-2h-4c0 1.1.9 2 2 2Zm-4-5h8v-6c0-2.48-1.51-4.5-4-4.5s-4 2.02-4 4.5v6Z",
  sizes: {
    small: {
      name: "notifications_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M13.5 11.911v-3.67c0-2.254-1.223-4.141-3.375-4.64v-.5C10.125 2.491 9.623 2 9 2s-1.125.492-1.125 1.101v.5C5.73 4.1 4.5 5.979 4.5 8.24v3.67L3 13.38v.673h12v-.673l-1.5-1.469ZM9 16.001c.825 0 1.5-.49 1.5-1.47h-3c0 .98.675 1.47 1.5 1.47Zm-3-3.356h6V8.24c0-1.82-1.133-3.303-3-3.303S6 6.42 6 8.24v4.405Z"
    }
  }
};
var notifications_active = {
  name: "notifications_active",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18 15.75v-5c0-3.07-1.63-5.64-4.5-6.32v-.68c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v.68C7.64 5.11 6 7.67 6 10.75v5l-2 2v1h16v-1l-2-2Zm-6 6c1.1 0 2-.9 2-2h-4c0 1.1.9 2 2 2Zm-4-5h8v-6c0-2.48-1.51-4.5-4-4.5s-4 2.02-4 4.5v6ZM7.58 3.83 6.15 2.4c-2.4 1.83-3.98 4.65-4.12 7.85h2a8.445 8.445 0 0 1 3.55-6.42Zm14.39 6.42h-2a8.495 8.495 0 0 0-3.54-6.42l1.42-1.43c2.39 1.83 3.97 4.65 4.12 7.85Z",
  sizes: {
    small: {
      name: "notifications_active_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M13.5 11.813v-3.75c0-2.303-1.223-4.23-3.375-4.74v-.51c0-.623-.503-1.126-1.125-1.126-.623 0-1.125.503-1.125 1.125v.51C5.73 3.832 4.5 5.752 4.5 8.063v3.75l-1.5 1.5V14h12v-.688l-1.5-1.5ZM9 16c1 0 1.5-.871 1.5-1.5h-3c0 .629.5 1.5 1.5 1.5Zm-3-3.438h6v-4.5c0-1.86-1.133-3.374-3-3.374-1.868 0-3 1.514-3 3.375v4.5Zm-.315-9.69L4.612 1.8a7.819 7.819 0 0 0-3.09 5.888h1.5a6.334 6.334 0 0 1 2.663-4.815Zm10.793 4.816h-1.5a6.371 6.371 0 0 0-2.655-4.815L13.386 1.8a7.866 7.866 0 0 1 3.09 5.888Z"
    }
  }
};
var notifications_off = {
  name: "notifications_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15.305 10.75c0-2.48-1.51-4.5-4-4.5-.144 0-.282.017-.42.034l-.13.016-1.64-1.64.105-.038c.188-.07.383-.141.585-.192v-.68c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5v.68c2.87.68 4.5 3.25 4.5 6.32v2.1l-2-2v-.1Zm-2 9c0 1.1-.9 2-2 2s-2-.9-2-2h4ZM4.715 3.1l-1.41 1.41 2.81 2.81c-.52 1-.81 2.17-.81 3.43v5l-2 2v1h14.24l1.74 1.74 1.41-1.41L4.715 3.1Zm2.59 13.65h8v-.24l-7.66-7.66c-.22.58-.34 1.22-.34 1.9v6Z",
  sizes: {
    small: {
      name: "notifications_off_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M11.47 8.236c0-1.82-1.129-3.302-2.99-3.302-.108 0-.21.013-.313.026l-.098.011-1.226-1.203.078-.028c.14-.051.287-.104.438-.14v-.5c0-.608.5-1.1 1.121-1.1.62 0 1.121.492 1.121 1.1v.5c2.146.498 3.364 2.384 3.364 4.636v1.54L11.47 8.31v-.073Zm-1.473 6.297c0 .978-.673 1.467-1.495 1.467-.822 0-1.495-.49-1.495-1.467h2.99ZM3.554 2.623 2.5 3.659l2.1 2.061a5.36 5.36 0 0 0-.605 2.517v3.668L2.5 13.37v.672h10.487l1.459 1.338 1.054-1.034L3.554 2.624ZM5.49 12.639h5.98v-.176l-5.726-5.62a3.856 3.856 0 0 0-.254 1.394v4.402Z"
    }
  }
};
var error_filled = {
  name: "error_filled",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2Zm-1 15v-2h2v2h-2Zm0-10v6h2V7h-2Z",
  sizes: {
    small: {
      name: "error_filled_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M9 2C5.136 2 2 5.136 2 9s3.136 7 7 7 7-3.136 7-7-3.136-7-7-7ZM8 13v-2h2v2H8Zm0-8v5h2V5H8Z"
    }
  }
};
var warning_filled = {
  name: "warning_filled",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m23 21.5-11-19-11 19h22Zm-12-3v-2h2v2h-2Zm0-4h2v-4h-2v4Z",
  sizes: {
    small: {
      name: "warning_filled_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M17.5 16 9 2 .5 16h17ZM8 14v-2h2v2H8Zm0-4h2V6H8v4Z"
    }
  }
};
var menu = {
  name: "menu",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 8V6h18v2H3Zm0 5h18v-2H3v2Zm0 5h18v-2H3v2Z"
};
var apps = {
  name: "apps",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 8h4V4H4v4Zm6 12h4v-4h-4v4Zm-2 0H4v-4h4v4Zm-4-6h4v-4H4v4Zm10 0h-4v-4h4v4Zm2-10v4h4V4h-4Zm-2 4h-4V4h4v4Zm2 6h4v-4h-4v4Zm4 6h-4v-4h4v4Z"
};
var home = {
  name: "home",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 12.5H2l10-9 10 9h-3v8h-6v-6h-2v6H5v-8Zm12-1.81-5-4.5-5 4.5v7.81h2v-6h6v6h2v-7.81Z"
};
var exit_to_app = {
  name: "exit_to_app",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 3h14c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H5a2 2 0 0 1-2-2v-4h2v4h14V5H5v4H3V5a2 2 0 0 1 2-2Zm6.5 14-1.41-1.41L12.67 13H3v-2h9.67l-2.58-2.59L11.5 7l5 5-5 5Z"
};
var open_in_browser = {
  name: "open_in_browser",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 4h14a2 2 0 0 1 2 2v12c0 1.1-.9 2-2 2h-4v-2h4V8H5v10h4v2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Zm3 10 4-4 4 4h-3v6h-2v-6H8Z"
};
var external_link = {
  name: "external_link",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 3c-1.093 0-2 .907-2 2v14c0 1.093.907 2 2 2h14c1.093 0 2-.907 2-2v-7h-2v7H5V5h7V3H5Zm9 0v2h3.586l-9.293 9.293 1.414 1.414L19 6.414V10h2V3h-7Z"
};
var category = {
  name: "category",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m6 11 5.5-9 5.5 9H6Zm7.43-2L11.5 5.84 9.56 9h3.87ZM17 13c-2.49 0-4.5 2.01-4.5 4.5S14.51 22 17 22s4.5-2.01 4.5-4.5S19.49 13 17 13Zm-2.5 4.5a2.5 2.5 0 0 0 5 0 2.5 2.5 0 0 0-5 0Zm-12 4h8v-8h-8v8Zm6-6h-4v4h4v-4Z"
};
var settings = {
  name: "settings",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19.502 12c0 .34-.03.66-.07.98l2.11 1.65c.19.15.24.42.12.64l-2 3.46c-.09.16-.26.25-.43.25-.06 0-.12-.01-.18-.03l-2.49-1c-.52.39-1.08.73-1.69.98l-.38 2.65c-.03.24-.24.42-.49.42h-4c-.25 0-.46-.18-.49-.42l-.38-2.65c-.61-.25-1.17-.58-1.69-.98l-2.49 1a.5.5 0 0 1-.61-.22l-2-3.46a.505.505 0 0 1 .12-.64l2.11-1.65a7.93 7.93 0 0 1-.07-.98c0-.33.03-.66.07-.98l-2.11-1.65a.493.493 0 0 1-.12-.64l2-3.46c.09-.16.26-.25.43-.25.06 0 .12.01.18.03l2.49 1c.52-.39 1.08-.73 1.69-.98l.38-2.65c.03-.24.24-.42.49-.42h4c.25 0 .46.18.49.42l.38 2.65c.61.25 1.17.58 1.69.98l2.49-1a.5.5 0 0 1 .61.22l2 3.46c.12.22.07.49-.12.64l-2.11 1.65c.04.32.07.64.07.98Zm-2 0c0-.21-.01-.42-.05-.73l-.14-1.13.89-.7 1.07-.85-.7-1.21-1.27.51-1.06.43-.91-.7c-.4-.3-.8-.53-1.23-.71l-1.06-.43-.16-1.13-.19-1.35h-1.39l-.2 1.35-.16 1.13-1.06.43c-.41.17-.82.41-1.25.73l-.9.68-1.04-.42-1.27-.51-.7 1.21 1.08.84.89.7-.14 1.13c-.03.3-.05.53-.05.73 0 .2.02.43.05.74l.14 1.13-.89.7-1.08.84.7 1.21 1.27-.51 1.06-.43.91.7c.4.3.8.53 1.23.71l1.06.43.16 1.13.19 1.35h1.4l.2-1.35.16-1.13 1.06-.43c.41-.17.82-.41 1.25-.73l.9-.68 1.04.42 1.27.51.7-1.21-1.08-.84-.89-.7.14-1.13c.03-.3.05-.52.05-.73Zm-5.5-4c-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4-1.79-4-4-4Zm-2 4c0 1.1.9 2 2 2s2-.9 2-2-.9-2-2-2-2 .9-2 2Z"
};
var launch = {
  name: "launch",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M6 15c-.83 0-1.58.34-2.12.88C2.7 17.06 2 22 2 22s4.94-.7 6.12-1.88A2.996 2.996 0 0 0 6 15Zm.71 3.71c-.28.28-2.17.76-2.17.76s.47-1.88.76-2.17c.17-.19.42-.3.7-.3a1.003 1.003 0 0 1 .71 1.71Zm10.71-5.06c6.36-6.36 4.24-11.31 4.24-11.31S16.71.22 10.35 6.58l-2.49-.5a2.03 2.03 0 0 0-1.81.55L2 10.69l5 2.14L11.17 17l2.14 5 4.05-4.05c.47-.47.68-1.15.55-1.81l-.49-2.49ZM7.41 10.83l-1.91-.82 1.97-1.97 1.44.29c-.57.83-1.08 1.7-1.5 2.5Zm6.58 7.67-.82-1.91c.8-.42 1.67-.93 2.49-1.5l.29 1.44-1.96 1.97ZM16 12.24c-1.32 1.32-3.38 2.4-4.04 2.73l-2.93-2.93c.32-.65 1.4-2.71 2.73-4.04 4.68-4.68 8.23-3.99 8.23-3.99s.69 3.55-3.99 8.23ZM15 11c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2Z"
};
var go_to = {
  name: "go_to",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2Zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8Zm.8-9.143V8l4.2 4-4.2 4v-2.857H7v-2.286h5.8Z"
};
var subsea_drone = {
  name: "subsea_drone",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16.9 4.18a.893.893 0 0 1 1.31 0l.064.068a2.979 2.979 0 0 0 4.179.19l.216-.194a1 1 0 1 0-1.338-1.487l-.216.195a.979.979 0 0 1-1.374-.063l-.063-.068a2.893 2.893 0 0 0-4.245 0 .893.893 0 0 1-1.31 0 2.893 2.893 0 0 0-4.246 0 .893.893 0 0 1-1.31 0 2.893 2.893 0 0 0-4.245 0l-.063.068a.979.979 0 0 1-1.374.063l-.216-.195a1 1 0 1 0-1.338 1.487l.216.194a2.979 2.979 0 0 0 4.18-.19l.062-.068a.893.893 0 0 1 1.31 0 2.893 2.893 0 0 0 4.246 0 .893.893 0 0 1 1.31 0 2.893 2.893 0 0 0 4.245 0ZM4 11h17V9H4a1 1 0 1 0 0 2Zm0 2h5.091l-2.777 4.011A1.5 1.5 0 1 0 7.618 19.5H14l2.06 2.06a1.5 1.5 0 0 0 1.061.44h1.258a1.5 1.5 0 0 0 1.06-.44L20.5 20.5 19 19l-1 1h-.5L16 18.5l1.5-1.5h.5l1 1 1.5-1.5-1.06-1.06a1.5 1.5 0 0 0-1.061-.44H17.12a1.5 1.5 0 0 0-1.06.44L14 17.5H8.409l3.115-4.5H21a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2H4a3 3 0 0 0 0 6Z"
};
var onshore_drone = {
  name: "onshore_drone",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5.5 6 4 7.5l1.06 1.06A1.5 1.5 0 0 0 6.122 9H7.38a1.5 1.5 0 0 0 1.06-.44L10.5 6.5h8.326L16.403 11H4a3 3 0 1 0 0 6h17a2 2 0 0 0 2-2v-2a2 2 0 0 0-2-2h-2.326l2.176-4.041A1.5 1.5 0 1 0 19.382 4.5H10.5L8.44 2.44A1.5 1.5 0 0 0 7.378 2H6.12a1.5 1.5 0 0 0-1.06.44L4 3.5 5.5 5l1-1H7l1.5 1.5L7 7h-.5l-1-1ZM21 15H4a1 1 0 1 1 0-2h17v2ZM5 22a2 2 0 1 1 0-4 2 2 0 0 1 0 4Zm5-2a2 2 0 1 0 4 0 2 2 0 0 0-4 0Zm9 2a2 2 0 1 1 0-4 2 2 0 0 1 0 4Z"
};
var van = {
  name: "van",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17 5H3a2 2 0 0 0-2 2v9h2c0 1.66 1.34 3 3 3s3-1.34 3-3h6c0 1.66 1.34 3 3 3s3-1.34 3-3h2v-5l-6-6Zm-2 2h1l3 3h-4V7Zm-2 0H9v3h4V7ZM3 7h4v3H3V7Zm1.75 9a1.25 1.25 0 1 0 2.5 0 1.25 1.25 0 0 0-2.5 0ZM18 17.25a1.25 1.25 0 1 1 0-2.5 1.25 1.25 0 0 1 0 2.5ZM20.22 14H21v-2H3v2h.78c.55-.61 1.33-1 2.22-1 .89 0 1.67.39 2.22 1h7.56c.55-.61 1.34-1 2.22-1 .88 0 1.67.39 2.22 1Z"
};
var motorcycle = {
  name: "motorcycle",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11 5h4.41l4.03 4.03C22.03 9.23 24 11.35 24 14c0 2.8-2.2 5-5 5s-5-2.2-5-5c0-.63.11-1.23.32-1.77L11.55 15H9.9c-.45 2.31-2.44 4-4.9 4-2.8 0-5-2.2-5-5s2.2-5 5-5h11.59l-2-2H11V5Zm-.28 8 2-2H8.98c.3.39.54.83.72 1.31l.25.69h.77ZM19 17c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3ZM2 14c0 1.63 1.37 3 3 3 1.28 0 2.4-.85 2.82-2H5v-2h2.82C7.4 11.85 6.28 11 5 11c-1.63 0-3 1.37-3 3Z"
};
var transit_enter_exit = {
  name: "transit_enter_exit",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16 18H6V8h3v4.77L15.98 6 18 8.03 11.15 15H16v3Z"
};
var trip_origin = {
  name: "trip_origin",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2Zm6 10c0 3.31-2.69 6-6 6s-6-2.69-6-6 2.69-6 6-6 6 2.69 6 6Z"
};
var satellite = {
  name: "satellite",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2ZM5 19V5h14v14H5ZM6 6h2.57c0 1.42-1.15 2.58-2.57 2.58V6Zm6 0h-1.71c0 2.36-1.92 4.29-4.29 4.29V12c3.32 0 6-2.69 6-6Zm-.86 9.73 3-3.87L18 17H6l3-3.85 2.14 2.58Z"
};
var traffic_light = {
  name: "traffic_light",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20 10h-3V8.86c1.72-.45 3-2 3-3.86h-3V4c0-.55-.45-1-1-1H8c-.55 0-1 .45-1 1v1H4c0 1.86 1.28 3.41 3 3.86V10H4c0 1.86 1.28 3.41 3 3.86V15H4c0 1.86 1.28 3.41 3 3.86V20c0 .55.45 1 1 1h8c.55 0 1-.45 1-1v-1.14c1.72-.45 3-2 3-3.86h-3v-1.14c1.72-.45 3-2 3-3.86Zm-5-5v14H9V5h6Zm-1.5 11.5c0 .83-.67 1.5-1.5 1.5s-1.5-.67-1.5-1.5.67-1.5 1.5-1.5 1.5.67 1.5 1.5Zm-1.5-3c.83 0 1.5-.67 1.5-1.5s-.67-1.5-1.5-1.5-1.5.67-1.5 1.5.67 1.5 1.5 1.5Zm1.5-6c0 .83-.67 1.5-1.5 1.5s-1.5-.67-1.5-1.5S11.17 6 12 6s1.5.67 1.5 1.5Z"
};
var hospital = {
  name: "hospital",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 3H5c-1.1 0-1.99.9-1.99 2L3 19c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2Zm0 16H5V5h14v14Zm-5.5-2h-3v-3.5H7v-3h3.5V7h3v3.5H17v3h-3.5V17Z"
};
var map = {
  name: "map",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20.34 3.03 20.5 3c.28 0 .5.22.5.5v15.12c0 .23-.15.41-.36.48L15 21l-6-2.1-5.34 2.07-.16.03c-.28 0-.5-.22-.5-.5V5.38c0-.23.15-.41.36-.48L9 3l6 2.1 5.34-2.07ZM14 6.87l-4-1.4v11.66l4 1.4V6.87Zm-9-.41 3-1.01v11.7l-3 1.16V6.46Zm11 12.09 3-1.01V5.7l-3 1.16v11.69Z"
};
var parking = {
  name: "parking",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5.5 3h7c3.31 0 6 2.69 6 6s-2.69 6-6 6h-3v6h-4V3Zm4 8h3.2c1.1 0 2-.9 2-2s-.9-2-2-2H9.5v4Z"
};
var directions = {
  name: "directions",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m22.427 10.593-9.01-9.01c-.75-.75-2.07-.76-2.83 0l-9 9c-.78.78-.78 2.04 0 2.82l9 9c.39.39.9.58 1.41.58.51 0 1.02-.19 1.41-.58l8.99-8.99c.79-.76.8-2.02.03-2.82Zm-10.42 10.4-9-9 9-9 9 9-9 9Zm-4.01-5.99v-4c0-.55.45-1 1-1h5v-2.5l3.5 3.5-3.5 3.5v-2.5h-4v3h-2Z"
};
var transfer = {
  name: "transfer",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M9.5 5.25c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2Zm6.99 8.25v1.75H22v1.5h-5.51v1.75L14 16l2.49-2.5Zm-2.49 6h5.51v-1.75l2.49 2.5-2.49 2.5V21H14v-1.5ZM3 22.75l2.75-14.1L4 9.4v3.35H2v-4.7L7.25 5.9c.25-.1.5-.15.75-.15.7 0 1.35.35 1.7.95l.95 1.6c.9 1.45 2.5 2.45 4.35 2.45v2c-2.2 0-4.15-1-5.45-2.6l-.6 3L11 15.2v7.55H9v-6l-2.15-2-1.75 8H3Z"
};
var terrain = {
  name: "terrain",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M9.78 11.63 14 6l9 12H1l5.53-7.37L10.54 16H19l-5-6.67-2.97 3.97-1.25-1.67Zm-3.26 2.34L5 16h3.04l-1.52-2.03Z"
};
var mall = {
  name: "mall",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17 6.5h2c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H5c-1.1 0-2-.9-2-2v-12c0-1.1.9-2 2-2h2c0-2.76 2.24-5 5-5s5 2.24 5 5Zm-2 0c0-1.66-1.34-3-3-3s-3 1.34-3 3h6Zm-10 14v-12h14v12H5Zm4-11c0 1.66 1.34 3 3 3s3-1.34 3-3h2c0 2.76-2.24 5-5 5s-5-2.24-5-5h2Z"
};
var ticket = {
  name: "ticket",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M22 10V6c0-1.1-.9-2-2-2H4c-1.1 0-1.99.9-1.99 2v4c1.1 0 1.99.9 1.99 2a2 2 0 0 1-2 2v4c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2v-4c-1.1 0-2-.9-2-2s.9-2 2-2Zm-2-1.46c-1.19.69-2 1.99-2 3.46s.81 2.77 2 3.46V18H4v-2.54c1.19-.69 2-1.99 2-3.46 0-1.48-.8-2.77-1.99-3.46L4 6h16v2.54Zm-8 5.58L9.07 16l.88-3.37-2.69-2.2 3.47-.21L12 7l1.26 3.23 3.47.21-2.69 2.2.89 3.36L12 14.12Z"
};
var pharmacy = {
  name: "pharmacy",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21 6h-2.64l1.14-3.14L17.15 2l-1.46 4H3v2l2 6-2 6v2h18v-2l-2-6 2-6V6Zm-3.9 8.63L18.89 20H5.11l1.79-5.37.21-.63-.21-.63L5.11 8h13.78l-1.79 5.37-.21.63.21.63ZM11 10h2v3h3v2h-3v3h-2v-3H8v-2h3v-3Z"
};
var cinema = {
  name: "cinema",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18 3h2v18h-2v-2h-2v2H8v-2H6v2H4V3h2v2h2V3h8v2h2V3Zm-4 16V5h-4v14h4Zm2-10V7h2v2h-2ZM6 7v2h2V7H6Zm10 6v-2h2v2h-2ZM6 11v2h2v-2H6Zm10 6v-2h2v2h-2ZM6 15v2h2v-2H6Z"
};
var convenience_store = {
  name: "convenience_store",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 4v3h3v13h-8v-4h-4v4H2V7h3V4h14Zm-3 14h4V9h-3V6H7v3H4v9h4v-4h8v4ZM8 8h2v1H8v3h3v-1H9v-1h2V7H8v1Zm6 1h1V7h1v5h-1v-2h-2V7h1v2Z"
};
var car_wash = {
  name: "car_wash",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 5.1c.83 0 1.5-.67 1.5-1.5C8.5 2.6 7 .9 7 .9S5.5 2.6 5.5 3.6c0 .83.67 1.5 1.5 1.5Zm6.5-1.5c0 .83-.67 1.5-1.5 1.5s-1.5-.67-1.5-1.5c0-1 1.5-2.7 1.5-2.7s1.5 1.7 1.5 2.7Zm5 0c0 .83-.67 1.5-1.5 1.5s-1.5-.67-1.5-1.5c0-1 1.5-2.7 1.5-2.7s1.5 1.7 1.5 2.7Zm-1 3.5c.66 0 1.22.42 1.42 1.01L21 14.1v8c0 .55-.45 1-1 1h-1c-.55 0-1-.45-1-1v-1H6v1c0 .55-.45 1-1 1H4c-.55 0-1-.45-1-1v-8l2.08-5.99c.21-.59.76-1.01 1.42-1.01h11Zm-10.65 2h10.29l1.04 3H5.81l1.04-3ZM5 14.44v4.66h14v-4.66l-.11-.34H5.12l-.12.34Zm2.5.66a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm7.5 1.5a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0Z"
};
var library = {
  name: "library",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16 5.225c0 2.21-1.79 4-4 4s-4-1.79-4-4 1.79-4 4-4 4 1.79 4 4Zm-2 0c0-1.1-.9-2-2-2s-2 .9-2 2 .9 2 2 2 2-.9 2-2Zm-2 6.55c-2.36-2.2-5.52-3.55-9-3.55v11c3.48 0 6.64 1.35 9 3.55 2.36-2.19 5.52-3.55 9-3.55v-11c-3.48 0-6.64 1.35-9 3.55Zm0 8.4c2.07-1.52 4.47-2.48 7-2.82v-6.95c-2.1.38-4.05 1.35-5.64 2.83L12 14.505l-1.36-1.28A11.18 11.18 0 0 0 5 10.395v6.95a15.2 15.2 0 0 1 7 2.83Z"
};
var store = {
  name: "store",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 4h16v2H4V4Zm14.96 8-.6-3H5.64l-.6 3h13.92ZM20 7H4l-1 5v2h1v6h10v-6h4v6h2v-6h1v-2l-1-5ZM6 14v4h6v-4H6Z"
};
var hotel = {
  name: "hotel",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 6.5h-8v8H3v-10H1v15h2v-3h18v3h2v-9c0-2.21-1.79-4-4-4Zm-9 4c0 1.66-1.34 3-3 3s-3-1.34-3-3 1.34-3 3-3 3 1.34 3 3Zm-2 0c0-.55-.45-1-1-1s-1 .45-1 1 .45 1 1 1 1-.45 1-1Zm5 4h8v-4c0-1.1-.9-2-2-2h-6v6Z"
};
var grocery_store = {
  name: "grocery_store",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16.546 13c.75 0 1.41-.41 1.75-1.03l3.58-6.49a.996.996 0 0 0-.87-1.48h-14.8l-.94-2h-3.27v2h2l3.6 7.59-1.35 2.44c-.73 1.34.23 2.97 1.75 2.97h12v-2h-12l1.1-2h7.45Zm-10.54 7c0-1.1.89-2 1.99-2s2 .9 2 2-.9 2-2 2-1.99-.9-1.99-2Zm10 0c0-1.1.89-2 1.99-2s2 .9 2 2-.9 2-2 2-1.99-.9-1.99-2Zm3.3-14H7.156l2.37 5h7.02l2.76-5Z"
};
var walk = {
  name: "walk",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15 3.25c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2Zm-8.5 19.5 2.8-14.1-1.8.7v3.4h-2v-4.7l5.05-2.14c.97-.41 2.09-.05 2.65.84l1 1.6c.8 1.4 2.4 2.4 4.3 2.4v2c-2.2 0-4.2-1-5.5-2.5l-.6 3 2.1 2v7.5h-2v-6l-2.1-2-1.8 8H6.5Z"
};
var run = {
  name: "run",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16.55 3.25c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2Zm-4.6 11.5-1 4.4-7-1.4.4-2 4.9 1 1.6-8.1-1.8.7v3.4h-2v-4.7l5.2-2.2c.15 0 .275-.025.4-.05s.25-.05.4-.05c.7 0 1.3.4 1.7 1l1 1.6c.8 1.4 2.4 2.4 4.3 2.4v2c-2.2 0-4.2-1-5.5-2.5l-.6 3 2.1 2v7.5h-2v-6l-2.1-2Z"
};
var bike = {
  name: "bike",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17.5 3.75c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2ZM0 17.25c0-2.8 2.2-5 5-5s5 2.2 5 5-2.2 5-5 5-5-2.2-5-5Zm5 3.5c-1.9 0-3.5-1.6-3.5-3.5s1.6-3.5 3.5-3.5 3.5 1.6 3.5 3.5-1.6 3.5-3.5 3.5Zm14.1-9.5c-2.1 0-3.8-.8-5.1-2.1l-.8-.8-2.4 2.4 2.2 2.3v6.2h-2v-5l-3.2-2.8c-.4-.3-.6-.8-.6-1.4 0-.5.2-1 .6-1.4l2.8-2.8c.3-.4.8-.6 1.4-.6.6 0 1.1.2 1.6.6l1.9 1.9c.9.9 2.1 1.5 3.6 1.5v2Zm-.1 1c-2.8 0-5 2.2-5 5s2.2 5 5 5 5-2.2 5-5-2.2-5-5-5Zm-3.5 5c0 1.9 1.6 3.5 3.5 3.5s3.5-1.6 3.5-3.5-1.6-3.5-3.5-3.5-3.5 1.6-3.5 3.5Z"
};
var boat = {
  name: "boat",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M9.001 1h6v3h3c1.1 0 2 .9 2 2v4.62l1.28.42c.26.08.48.26.6.5s.14.52.06.78L20.051 19h-.05c-1.6 0-3.02-.88-4-2-.98 1.12-2.4 2-4 2s-3.02-.88-4-2c-.98 1.12-2.4 2-4 2h-.05l-1.9-6.68a1.007 1.007 0 0 1 .66-1.28l1.29-.42V6c0-1.1.9-2 2-2h3V1Zm4 2v1h-2V3h2Zm-1 7.11 5.38 1.77 2.39.78-1.12 3.97c-.54-.3-.94-.71-1.14-.94l-1.51-1.73-1.51 1.72c-.34.4-1.28 1.32-2.49 1.32-1.21 0-2.15-.92-2.49-1.32l-1.51-1.72-1.51 1.72c-.2.23-.6.63-1.14.93l-1.13-3.96 2.4-.79 5.38-1.75Zm-6-.14V6h12v3.97l-6-1.97-6 1.97Zm6 10.99c1.39 0 2.78-.43 4-1.28 1.22.85 2.61 1.32 4 1.32h2v2h-2c-1.38 0-2.74-.34-4-.99a8.71 8.71 0 0 1-4 .97c-1.37 0-2.74-.33-4-.97-1.26.64-2.62.99-4 .99h-2v-2h2c1.39 0 2.78-.47 4-1.32 1.22.85 2.61 1.28 4 1.28Z"
};
var place_unknown = {
  name: "place_unknown",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 8.755c0-4.41 3.59-8 8-8s8 3.59 8 8c0 5.57-6.96 13.34-7.26 13.67l-.74.82-.74-.82c-.3-.33-7.26-8.1-7.26-13.67Zm2 0c0 3.54 3.82 8.86 6 11.47 1.75-2.11 6-7.64 6-11.47 0-3.31-2.69-6-6-6s-6 2.69-6 6Zm5.13 5h1.75v1.75h-1.75v-1.75Zm-2.63-5.5c0-1.93 1.57-3.5 3.5-3.5s3.5 1.57 3.5 3.5c0 1.124-.69 1.729-1.362 2.318-.637.559-1.258 1.103-1.258 2.062h-1.75c0-1.59.82-2.22 1.543-2.776.569-.438 1.077-.829 1.077-1.604 0-.96-.79-1.75-1.75-1.75s-1.75.79-1.75 1.75H8.5Z"
};
var flight = {
  name: "flight",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21.5 16v-2l-8-5V3.5c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5l-2-1.5v-5.5l8 2.5Z"
};
var subway_tunnel = {
  name: "subway_tunnel",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2c1.86 0 4 .09 5.8.8C20.47 3.84 22 6.05 22 8.86V22H2V8.86C2 6.05 3.53 3.84 6.2 2.8 8 2.09 10.14 2 12 2Zm-1.33 16.5L9.17 20h5.66l-1.5-1.5h-2.66ZM7.01 14V9h10v5h-10Zm8.49 3c.55 0 1-.45 1-1s-.45-1-1-1-1 .45-1 1 .45 1 1 1Zm-7-2c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1Zm8 5H20V8.86c0-2-1.01-3.45-2.93-4.2C15.59 4.08 13.68 4 12 4c-1.68 0-3.59.08-5.07.66C5.01 5.41 4 6.86 4 8.86V20h3.5v-.38l1.15-1.16A2.979 2.979 0 0 1 6 15.5V9c0-2.63 3-3 6-3s6 .37 6 3v6.5c0 1.54-1.16 2.79-2.65 2.96l1.15 1.16V20Z"
};
var tram = {
  name: "tram",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m13 5 .75-1.5H17V2H7v1.5h4.75L11 5c-3.13.09-6 .73-6 3.5V17c0 1.5 1.11 2.73 2.55 2.95L6 21.5v.5h2l2-2h4l2 2h2v-.5l-1.55-1.55C17.89 19.73 19 18.5 19 17V8.5c0-2.77-2.87-3.41-6-3.5Zm-1.97 2h1.94c2.75.08 3.62.58 3.9 1H7.13c.28-.42 1.15-.92 3.9-1ZM7.74 17.95h3.11c-.22-.26-.35-.59-.35-.95 0-.39.15-.73.39-1H7v1c0 .45.3.84.74.95ZM17 17c0 .45-.3.84-.74.95h-3.11c.22-.26.35-.59.35-.95 0-.39-.15-.73-.39-1H17v1ZM7 14h10v-4H7v4Z"
};
var train = {
  name: "train",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 6.5c0-3.5 4-4 8-4s8 .5 8 4V16c0 1.93-1.57 3.5-3.5 3.5L18 21v.5h-2l-2-2h-4l-2 2H6V21l1.5-1.5C5.57 19.5 4 17.93 4 16V6.5Zm4.5 7a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm7 0a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm-3.5-9c3.51 0 4.96.48 5.57 1H6.43c.61-.52 2.06-1 5.57-1Zm-1 3H6v3h5v-3Zm7 8.5c0 .83-.67 1.5-1.5 1.5h-9c-.83 0-1.5-.67-1.5-1.5v-3.5h12V16Zm-5-5.5h5v-3h-5v3Z"
};
var shipping = {
  name: "shipping",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20 8h-3V4H3c-1.1 0-2 .9-2 2v11h2c0 1.66 1.34 3 3 3s3-1.34 3-3h6c0 1.66 1.34 3 3 3s3-1.34 3-3h2v-5l-3-4Zm-.5 1.5 1.96 2.5H17V9.5h2.5ZM5 17c0 .55.45 1 1 1s1-.45 1-1-.45-1-1-1-1 .45-1 1Zm3.22-2c-.55-.61-1.33-1-2.22-1-.89 0-1.67.39-2.22 1H3V6h12v9H8.22ZM17 17c0 .55.45 1 1 1s1-.45 1-1-.45-1-1-1-1 .45-1 1Z"
};
var taxi = {
  name: "taxi",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18.92 6.01C18.72 5.42 18.16 5 17.5 5H15V3H9v2H6.5c-.66 0-1.21.42-1.42 1.01L3 12v8c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-1h12v1c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-8l-2.08-5.99ZM6.85 7h10.29l1.04 3H5.81l1.04-3ZM5 17h14v-4.66l-.11-.34H5.12l-.12.34V17Zm2.5-4a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm7.5 1.5a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0Z"
};
var transit = {
  name: "transit",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2.5c-4 0-8 .5-8 4V16c0 1.93 1.57 3.5 3.5 3.5L6 21v.5h12V21l-1.5-1.5c1.93 0 3.5-1.57 3.5-3.5V6.5c0-3.5-3.58-4-8-4Zm5.66 3H6.43c.61-.52 2.06-1 5.57-1 3.71 0 5.12.46 5.66 1Zm-6.66 5v-3H6v3h5Zm2-3h5v3h-5v-3ZM6 16c0 .83.67 1.5 1.5 1.5h9c.83 0 1.5-.67 1.5-1.5v-3.5H6V16Zm2.5-2.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3ZM14 15a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0Z"
};
var subway = {
  name: "subway",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2.5c-4 0-8 .5-8 4V16c0 1.93 1.57 3.5 3.5 3.5L6 21v.5h12V21l-1.5-1.5c1.93 0 3.5-1.57 3.5-3.5V6.5c0-3.5-3.58-4-8-4Zm5.66 3H6.43c.61-.52 2.06-1 5.57-1 3.71 0 5.12.46 5.66 1Zm-6.66 5v-3H6v3h5Zm2-3h5v3h-5v-3ZM6 16c0 .83.67 1.5 1.5 1.5h9c.83 0 1.5-.67 1.5-1.5v-3.5H6V16Zm2.5-2.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3ZM14 15a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0Z"
};
var car = {
  name: "car",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18.92 5.01C18.72 4.42 18.16 4 17.5 4h-11c-.66 0-1.21.42-1.42 1.01L3 11v8c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-1h12v1c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-8l-2.08-5.99ZM6.85 6h10.29l1.08 3.11H5.77L6.85 6ZM5 16h14v-5H5v5Zm2.5-4a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm7.5 1.5a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0Z"
};
var railway = {
  name: "railway",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2c-4.42 0-8 .5-8 4v10.5C4 18.43 5.57 20 7.5 20L6 21.5v.5h12v-.5L16.5 20c1.93 0 3.5-1.57 3.5-3.5V6c0-3.5-3.58-4-8-4Zm0 2c6 0 6 1.2 6 2H6c0-.8 0-2 6-2Zm6 7V8H6v3h12ZM7.5 18c-.83 0-1.5-.67-1.5-1.5V13h12v3.5c0 .83-.67 1.5-1.5 1.5h-9Zm2.5-2.5c0-1.1.9-2 2-2s2 .9 2 2-.9 2-2 2-2-.9-2-2Z"
};
var bus = {
  name: "bus",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 6.5c0-3.5 3.58-4 8-4s8 .5 8 4v10c0 .88-.39 1.67-1 2.22v1.78c0 .55-.45 1-1 1h-1c-.55 0-1-.45-1-1v-1H8v1c0 .55-.45 1-1 1H6c-.55 0-1-.45-1-1v-1.78c-.61-.55-1-1.34-1-2.22v-10Zm8-2c-3.69 0-5.11.46-5.66.99h11.32c-.55-.53-1.97-.99-5.66-.99Zm6 2.99v3.01H6V7.49h12Zm-.63 10.01.29-.27c.13-.11.34-.36.34-.73v-4H6v4c0 .37.21.62.34.73l.29.27h10.74Zm-8.87-4a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3ZM14 15a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0Z"
};
var departure_board = {
  name: "departure_board",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16 1a6.98 6.98 0 0 0-5.75 3.02C9.84 4.01 9.43 4 9 4c-4.42 0-8 .5-8 4v10c0 .88.39 1.67 1 2.22V22c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-1h8v1c0 .55.45 1 1 1h1c.55 0 1-.45 1-1v-1.78c.61-.55 1-1.34 1-2.22v-3.08c3.39-.49 6-3.39 6-6.92 0-3.87-3.13-7-7-7ZM4 16.5a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0Zm7 0a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0ZM9.29 6H9c-3.69 0-5.11.46-5.66.99h5.74c.05-.33.12-.67.21-.99ZM3 8.99h6.08c.16 1.11.57 2.13 1.18 3.01H3V8.99Zm11.66 9.74c.13-.11.34-.36.34-.73v-3.08c-.94-.13-1.81-.45-2.59-.92H3v4c0 .37.21.62.34.73l.29.27h10.74l.29-.27ZM16 13c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5Zm-1-9h1.5v4.25l2.87 1.68-.75 1.23L15 9V4Z"
};
var place_edit = {
  name: "place_edit",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 8.755c0-4.41 3.59-8 8-8s8 3.59 8 8c0 5.57-6.96 13.34-7.26 13.67l-.74.82-.74-.82c-.3-.33-7.26-8.1-7.26-13.67Zm2 0c0 3.54 3.82 8.86 6 11.47 1.75-2.11 6-7.64 6-11.47 0-3.31-2.69-6-6-6s-6 2.69-6 6Zm2.51 2.05v1.44h1.44l3.92-3.93-1.43-1.43-3.93 3.92Zm5.34-5.34a.38.38 0 0 1 .54 0l.9.9c.15.15.15.39 0 .54l-.7.7-1.44-1.44.7-.7Z"
};
var place_add = {
  name: "place_add",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C8.14 2 5 5.14 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.86-3.14-7-7-7ZM7 9c0-2.76 2.24-5 5-5s5 2.24 5 5c0 2.88-2.88 7.19-5 9.88C9.92 16.21 7 11.85 7 9Zm4-1V6h2v2h2v2h-2v2h-2v-2H9V8h2Z"
};
var place_person = {
  name: "place_person",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 8.755c0-4.41 3.59-8 8-8s8 3.59 8 8c0 5.57-6.96 13.34-7.26 13.67l-.74.82-.74-.82c-.3-.33-7.26-8.1-7.26-13.67Zm2 0c0 3.54 3.82 8.86 6 11.47 1.75-2.11 6-7.64 6-11.47 0-3.31-2.69-6-6-6s-6 2.69-6 6Zm6 0c.83 0 1.5-.67 1.5-1.5s-.67-1.5-1.5-1.5-1.5.68-1.5 1.5c0 .83.67 1.5 1.5 1.5Zm-3 2.5c0-1 2-1.5 3-1.5s3 .5 3 1.5v.12c-.73.84-1.8 1.38-3 1.38s-2.27-.54-3-1.38v-.12Z"
};
var pin_drop = {
  name: "pin_drop",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2c3.31 0 6 2.69 6 6 0 4.5-6 11-6 11S6 12.5 6 8c0-3.31 2.69-6 6-6Zm7 20v-2H5v2h14ZM8 8c0-2.21 1.79-4 4-4s4 1.79 4 4c0 2.13-2.08 5.46-4 7.91-1.92-2.44-4-5.78-4-7.91Zm2 0c0-1.1.9-2 2-2s2 .9 2 2a2 2 0 1 1-4 0Z"
};
var place = {
  name: "place",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7ZM7 9c0-2.76 2.24-5 5-5s5 2.24 5 5c0 2.88-2.88 7.19-5 9.88C9.92 16.21 7 11.85 7 9Zm2.5 0a2.5 2.5 0 1 1 5 0 2.5 2.5 0 0 1-5 0Z"
};
var view_360 = {
  name: "view_360",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 5.5c-5.52 0-10 2.24-10 5 0 2.24 2.94 4.13 7 4.77v3.23l4-4-4-4v2.73c-3.15-.56-5-1.9-5-2.73 0-1.06 3.04-3 8-3s8 1.94 8 3c0 .73-1.46 1.89-4 2.53v2.05c3.53-.77 6-2.53 6-4.58 0-2.76-4.48-5-10-5Z"
};
var gps_fixed = {
  name: "gps_fixed",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20.94 11A8.994 8.994 0 0 0 13 3.06V1h-2v2.06A8.994 8.994 0 0 0 3.06 11H1v2h2.06A8.994 8.994 0 0 0 11 20.94V23h2v-2.06A8.994 8.994 0 0 0 20.94 13H23v-2h-2.06ZM12 8c-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4-1.79-4-4-4Zm-7 4c0 3.87 3.13 7 7 7s7-3.13 7-7-3.13-7-7-7-7 3.13-7 7Z"
};
var gps_not_fixed = {
  name: "gps_not_fixed",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M13 3.06c4.17.46 7.48 3.77 7.94 7.94H23v2h-2.06A8.994 8.994 0 0 1 13 20.94V23h-2v-2.06A8.994 8.994 0 0 1 3.06 13H1v-2h2.06A8.994 8.994 0 0 1 11 3.06V1h2v2.06ZM5 12c0 3.87 3.13 7 7 7s7-3.13 7-7-3.13-7-7-7-7 3.13-7 7Z"
};
var gps_off = {
  name: "gps_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20.94 11A8.994 8.994 0 0 0 13 3.06V1h-2v2.06c-.98.11-1.91.38-2.77.78l1.53 1.53a6.995 6.995 0 0 1 8.87 8.87l1.53 1.53c.4-.86.67-1.79.78-2.77H23v-2h-2.06ZM3 4.27l2.04 2.04A8.994 8.994 0 0 0 3.06 11H1v2h2.06A8.994 8.994 0 0 0 11 20.94V23h2v-2.06c1.77-.2 3.38-.91 4.69-1.98L19.73 21l1.41-1.41L4.41 2.86 3 4.27ZM12 19c1.61 0 3.09-.55 4.27-1.46L6.46 7.73A6.995 6.995 0 0 0 12 19Z"
};
var near_me = {
  name: "near_me",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 10.53 21 3l-7.54 18h-.98l-2.64-6.84L3 11.51v-.98Zm10.03 6.33 4.24-10.13-10.13 4.23 3.43 1.33.82.32.32.83 1.32 3.42Z"
};
var navigation = {
  name: "navigation",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4.5 20.79 12 2.5l7.5 18.29-.71.71-6.79-3-6.79 3-.71-.71Zm11.78-2.59L12 7.77 7.72 18.2l3.47-1.53.81-.36.81.36 3.47 1.53Z"
};
var compass_calibration = {
  name: "compass_calibration",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2.5c-3.9 0-7.44 1.59-10 4.15l5 5a7.06 7.06 0 0 1 10-.01l5-5C19.44 4.09 15.9 2.5 12 2.5Zm-5 14c0-2.76 2.24-5 5-5s5 2.24 5 5-2.24 5-5 5-5-2.24-5-5Zm2 0c0 1.65 1.35 3 3 3s3-1.35 3-3-1.35-3-3-3-3 1.35-3 3Zm3-8.93c1.74 0 3.4.49 4.84 1.4l2.21-2.21A12.037 12.037 0 0 0 12 4.5c-2.56 0-5.01.79-7.06 2.26l2.21 2.22A8.973 8.973 0 0 1 12 7.57Z"
};
var flight_land = {
  name: "flight_land",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21.18 15.29c-.22.8-1.04 1.27-1.84 1.06L2.77 11.91V6.74l1.45.39.93 2.32 4.97 1.33V2.5l1.93.51 2.76 9.02 5.31 1.42c.8.22 1.27 1.04 1.06 1.84Zm.32 4.21h-19v2h19v-2Z"
};
var flight_takeoff = {
  name: "flight_takeoff",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20.25 8.295c.8-.22 1.63.26 1.84 1.06.21.8-.26 1.62-1.07 1.85l-16.57 4.43-2.59-4.49 1.45-.39 1.97 1.54 4.97-1.33-4.14-7.17 1.93-.51 6.9 6.43 5.31-1.42Zm1.27 10.42h-19v2h19v-2Z"
};
var commute = {
  name: "commute",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 4h7c1.66 0 3 1.34 3 3v1h-2V6H4v7h5v5H7l-2 2H4v-1l1-1c-1.66 0-3-1.34-3-3V7c0-1.66 1.34-3 3-3Zm1 11c0-.55-.45-1-1-1s-1 .45-1 1 .45 1 1 1 1-.45 1-1Zm14.57-5.34c-.14-.4-.52-.66-.97-.66h-7.19c-.46 0-.83.26-.98.66l-1.42 4.11v5.51c0 .38.31.72.69.72h.62c.38 0 .68-.38.68-.76V18h8v1.24c0 .38.31.76.69.76h.61c.38 0 .69-.34.69-.72l.01-1.37v-4.14l-1.43-4.11Zm-.97.34h-7.19l-1.03 3h9.25l-1.03-3ZM12 16c-.55 0-1-.45-1-1s.45-1 1-1 1 .45 1 1-.45 1-1 1Zm7-1c0 .55.45 1 1 1s1-.45 1-1-.45-1-1-1-1 .45-1 1Z"
};
var explore = {
  name: "explore",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10-4.48 10-10 10S2 17.52 2 12Zm2 0c0 4.41 3.59 8 8 8s8-3.59 8-8-3.59-8-8-8-8 3.59-8 8Zm2.5 5.5 7.51-3.49L17.5 6.5 9.99 9.99 6.5 17.5Zm6.6-5.5c0-.61-.49-1.1-1.1-1.1-.61 0-1.1.49-1.1 1.1 0 .61.49 1.1 1.1 1.1.61 0 1.1-.49 1.1-1.1Z"
};
var explore_off = {
  name: "explore_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20 12c0-4.41-3.59-8-8-8-1.48 0-2.86.41-4.05 1.12L6.49 3.66A9.91 9.91 0 0 1 12 2c5.52 0 10 4.48 10 10 0 2.04-.61 3.93-1.66 5.52l-1.46-1.46C19.59 14.86 20 13.48 20 12Zm-2.5-5.5-2.59 5.58-2.99-2.99L17.5 6.5ZM2.1 4.93l1.56 1.56A9.91 9.91 0 0 0 2 12c0 5.52 4.48 10 10 10 2.04 0 3.93-.61 5.51-1.66l1.56 1.56 1.41-1.41L3.51 3.51 2.1 4.93Zm7 6.99L5.12 7.94A7.932 7.932 0 0 0 4 12c0 4.41 3.59 8 8 8 1.48 0 2.86-.41 4.06-1.11l-3.98-3.98L6.5 17.5l2.6-5.58Z"
};
var toll = {
  name: "toll",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15 4c-4.42 0-8 3.58-8 8s3.58 8 8 8 8-3.58 8-8-3.58-8-8-8Zm0 14c-3.31 0-6-2.69-6-6s2.69-6 6-6 6 2.69 6 6-2.69 6-6 6ZM7 6.35a5.99 5.99 0 0 0 0 11.3v2.09c-3.45-.89-6-4.01-6-7.74 0-3.73 2.55-6.85 6-7.74v2.09Z"
};
var thumb_pin = {
  name: "thumb_pin",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m20.212 10.216-6.37-6.441-1.394 1.189 1.294 1.288-1.493 1.388-1.096 1.09-.497.495-.697.1-4.38.396H5.18l-.2.198 9.06 9.018.198-.198V18.44l.3-4.459.099-.793.597-.495 1.095-1.09 1.394-1.289 1.194 1.19 1.294-1.289Zm-9.258-6.342L13.841 1 23 10.117l-2.887 2.874a1.84 1.84 0 0 1-1.195.495 1.84 1.84 0 0 1-1.194-.495l-1.095 1.09-.299 4.46c0 .594-.2 1.189-.597 1.585l-1.693 1.685-5.226-5.203-6.122 6.095c-.199.198-.398.297-.697.297-.298 0-.497-.099-.696-.297a.955.955 0 0 1 0-1.388l6.122-6.094-5.326-5.302 1.692-1.685a1.97 1.97 0 0 1 1.394-.594h.199l4.48-.298 1.094-1.09c-.696-.594-.696-1.684 0-2.378Z"
};
var anchor = {
  name: "anchor",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12.909 9.158A3.638 3.638 0 0 0 11.999 2a3.636 3.636 0 0 0-.908 7.158v1.024H8.363V12h2.728v8.106a5.456 5.456 0 0 1-4.546-5.379H4.727a7.273 7.273 0 0 0 14.546 0h-1.819a5.456 5.456 0 0 1-4.545 5.38V12h2.727v-1.818H12.91V9.158Zm.909-3.522a1.818 1.818 0 1 1-3.636 0 1.818 1.818 0 0 1 3.636 0Z"
};
var aerial_drone = {
  name: "aerial_drone",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 5a1 1 0 1 0 2 0h1a1 1 0 1 0 0-2h-1a1 1 0 1 0-2 0H5a1 1 0 0 0-2 0H2a1 1 0 0 0 0 2h1a1 1 0 0 0 2 0h2.204L9.03 8.032A2 2 0 0 0 10.743 9h2.836a2 2 0 0 0 1.761-1.053L16.925 5H19ZM9.539 5h5.115l-1.075 2h-2.836L9.54 5Zm6.42 4.96c-2.234 2.234-5.684 2.234-7.918 0l-1.415 1.414c3.016 3.016 7.732 3.016 10.748 0L15.96 9.96ZM5.375 12.626c3.734 3.734 9.518 3.734 13.252 0l1.414 1.414c-4.515 4.516-11.565 4.516-16.08 0l1.414-1.414Zm15.919 2.667c-5.235 5.234-13.351 5.234-18.586 0l-1.414 1.414c6.015 6.016 15.399 6.016 21.414 0l-1.414-1.414Z"
};
var hand_radio = {
  name: "hand_radio",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M10 1a1 1 0 0 0-1 1v4H8a2 2 0 0 0-2 2v4a2 2 0 0 0 2 2v7a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-7a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-1V4a1 1 0 1 0-2 0v2h-2V2a1 1 0 0 0-1-1Zm0 13h4v7h-4v-7Zm6-6H8v4h8V8Zm-3 7h-2v1h2v-1Zm0 2h-2v1h2v-1Zm-2 2h2v1h-2v-1Z"
};
var camera = {
  name: "camera",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16.83 5H20c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V7c0-1.1.9-2 2-2h3.17L9 3h6l1.83 2ZM4 19h16V7h-4.05l-.59-.65L14.12 5H9.88L8.64 6.35 8.05 7H4v12Zm8-11c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5Zm-3.2 5c0 1.77 1.43 3.2 3.2 3.2 1.77 0 3.2-1.43 3.2-3.2 0-1.77-1.43-3.2-3.2-3.2-1.77 0-3.2 1.43-3.2 3.2Z"
};
var phone = {
  name: "phone",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 3h3.5c.55 0 1 .45 1 1 0 1.25.2 2.45.57 3.57.11.35.03.74-.25 1.02l-2.2 2.2c1.44 2.83 3.76 5.14 6.59 6.59l2.2-2.2c.2-.19.45-.29.71-.29.1 0 .21.01.31.05 1.12.37 2.33.57 3.57.57.55 0 1 .45 1 1V20c0 .55-.45 1-1 1-9.39 0-17-7.61-17-17 0-.55.45-1 1-1Zm2.54 2c.06.89.21 1.76.45 2.59l-1.2 1.2c-.41-1.2-.67-2.47-.76-3.79h1.51Zm9.86 12.02c.85.24 1.72.39 2.6.45v1.49c-1.32-.09-2.59-.35-3.8-.75l1.2-1.19Z"
};
var wifi = {
  name: "wifi",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m1 8.776 2 2c4.97-4.97 13.03-4.97 18 0l2-2c-6.07-6.07-15.92-6.07-22 0Zm8 8 3 3 3-3a4.237 4.237 0 0 0-6 0Zm-2-2-2-2c3.87-3.86 10.14-3.86 14 0l-2 2a7.074 7.074 0 0 0-10 0Z"
};
var wifi_off = {
  name: "wifi_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m2 4.22 1.41-1.41 16.97 16.97-1.41 1.41-7.08-7.08c-1.78.02-3.54.71-4.89 2.06l-2-2a9.823 9.823 0 0 1 4.41-2.54L7.17 9.39A12.65 12.65 0 0 0 3 12.17l-2-2C2.22 8.96 3.59 8 5.05 7.27L2 4.22Zm21 5.95-2 2a12.747 12.747 0 0 0-9.12-3.73L9.3 5.86c4.83-.84 9.97.58 13.7 4.31Zm-7.72 1.67c1.36.48 2.64 1.25 3.72 2.33l-.7.69-3.02-3.02ZM9 18.17l3 3 3-3a4.237 4.237 0 0 0-6 0Z"
};
var usb = {
  name: "usb",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15.1 7.4v4h1v2h-3v-8h2l-3-4-3 4h2v8h-3v-2.07c.7-.37 1.2-1.08 1.2-1.93 0-1.21-.99-2.2-2.2-2.2-1.21 0-2.2.99-2.2 2.2 0 .85.5 1.56 1.2 1.93v2.07c0 1.11.89 2 2 2h3v3.05c-.71.37-1.2 1.1-1.2 1.95a2.2 2.2 0 0 0 4.4 0c0-.85-.49-1.58-1.2-1.95V15.4h3c1.11 0 2-.89 2-2v-2h1v-4h-4Z"
};
var bluetooth = {
  name: "bluetooth",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18.355 7.71 12.645 2h-1v7.59L7.055 5l-1.41 1.41 5.59 5.59-5.59 5.59L7.055 19l4.59-4.59V22h1l5.71-5.71-4.3-4.29 4.3-4.29Zm-4.71-1.88 1.88 1.88-1.88 1.88V5.83Zm0 12.34 1.88-1.88-1.88-1.88v3.76Z"
};
var bluetooth_connected = {
  name: "bluetooth_connected",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17.71 7.71 12 2h-1v7.59L6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 11 14.41V22h1l5.71-5.71-4.3-4.29 4.3-4.29ZM7 12l-2-2-2 2 2 2 2-2Zm7.88-4.29L13 5.83v3.76l1.88-1.88Zm0 8.58L13 18.17v-3.76l1.88 1.88ZM17 12l2-2 2 2-2 2-2-2Z"
};
var bluetooth_disabled = {
  name: "bluetooth_disabled",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m13 5.83 1.88 1.88-1.6 1.6 1.41 1.41 3.02-3.02L12 2h-1v5.03l2 2v-3.2ZM5.41 4 4 5.41 10.59 12 5 17.59 6.41 19 11 14.41V22h1l4.29-4.29 2.3 2.29L20 18.59 5.41 4Zm9.47 12.29L13 14.41v3.76l1.88-1.88Z"
};
var bluetooth_searching = {
  name: "bluetooth_searching",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m9.995 2 5.71 5.71-4.3 4.29 4.3 4.29L9.995 22h-1v-7.59L4.405 19l-1.41-1.41L8.585 12l-5.59-5.59L4.405 5l4.59 4.59V2h1Zm9.53 4.71-1.26 1.26c.63 1.21.98 2.57.98 4.02 0 1.45-.36 2.82-.98 4.02l1.2 1.2a9.936 9.936 0 0 0 1.54-5.31c-.01-1.89-.55-3.67-1.48-5.19Zm-5.29 5.3 2.32 2.32c.28-.72.44-1.51.44-2.33 0-.82-.16-1.59-.43-2.31l-2.33 2.32Zm-3.24-6.18 1.88 1.88-1.88 1.88V5.83Zm0 12.34 1.88-1.88-1.88-1.88v3.76Z"
};
var apple_airplay = {
  name: "apple_airplay",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 2.5h18c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2h-4v-2h4v-12H3v12h4v2H3c-1.1 0-2-.9-2-2v-12c0-1.1.9-2 2-2Zm9 13 6 6H6l6-6Z"
};
var sim_card = {
  name: "sim_card",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M10 2h8c1.1 0 2 .9 2 2v16c0 1.1-.9 2-2 2H6c-1.1 0-2-.9-2-2V8l6-6Zm8 18V4h-7.17L6 8.83V20h12ZM7 17h2v2H7v-2Zm10 0h-2v2h2v-2ZM7 11h2v4H7v-4Zm6 4h-2v4h2v-4Zm-2-4h2v2h-2v-2Zm6 0h-2v4h2v-4Z"
};
var scanner = {
  name: "scanner",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m4.2 4.5 15.6 5.7c.7.2 1.2 1 1.2 1.8v5.5c0 1.1-.9 2-2 2H5c-1.1 0-2-.9-2-2v-4c0-1.1.9-2 2-2h12.6L3.5 6.4l.7-1.9Zm.8 13h14v-4H5v4Zm1-3h2v2H6v-2Zm12 0h-8v2h8v-2Z"
};
var router = {
  name: "router",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16 4.2c1.5 0 3 .6 4.2 1.7l.8-.8C19.6 3.7 17.8 3 16 3c-1.8 0-3.6.7-5 2.1l.8.8C13 4.8 14.5 4.2 16 4.2Zm-3.3 2.5.8.8c.7-.7 1.6-1 2.5-1 .9 0 1.8.3 2.5 1l.8-.8c-.9-.9-2.1-1.4-3.3-1.4-1.2 0-2.4.5-3.3 1.4ZM17 13h2c1.1 0 2 .9 2 2v4c0 1.1-.9 2-2 2H5c-1.1 0-2-.9-2-2v-4c0-1.1.9-2 2-2h10V9h2v4Zm2 6H5v-4h14v4ZM8 16H6v2h2v-2Zm1.5 0h2v2h-2v-2Zm5.5 0h-2v2h2v-2Z"
};
var memory = {
  name: "memory",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21 11V9h-2V7c0-1.1-.9-2-2-2h-2V3h-2v2h-2V3H9v2H7c-1.1 0-2 .9-2 2v2H3v2h2v2H3v2h2v2c0 1.1.9 2 2 2h2v2h2v-2h2v2h2v-2h2c1.1 0 2-.9 2-2v-2h2v-2h-2v-2h2ZM9 9h6v6H9V9Zm2 4h2v-2h-2v2Zm-4 4h10V7H7v10Z"
};
var headset_mic = {
  name: "headset_mic",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 10a9 9 0 0 1 18 0v10c0 1.66-1.34 3-3 3h-6v-2h7v-1h-4v-8h4v-2c0-3.87-3.13-7-7-7s-7 3.13-7 7v2h4v8H6c-1.66 0-3-1.34-3-3v-7Zm4 4v4H6c-.55 0-1-.45-1-1v-3h2Zm12 0v4h-2v-4h2Z"
};
var headset = {
  name: "headset",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 11.5a9 9 0 0 1 18 0v7c0 1.66-1.34 3-3 3h-3v-8h4v-2c0-3.87-3.13-7-7-7s-7 3.13-7 7v2h4v8H6c-1.66 0-3-1.34-3-3v-7Zm4 4v4H6c-.55 0-1-.45-1-1v-3h2Zm12 0v3c0 .55-.45 1-1 1h-1v-4h2Z"
};
var gamepad = {
  name: "gamepad",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15 2H9v5.5l3 3 3-3V2Zm-2 4.67V4h-2v2.67l1 1 1-1ZM17.33 13H20v-2h-2.67l-1 1 1 1ZM6.67 11l1 1-1 1H4v-2h2.67ZM13 17.33l-1-1-1 1V20h2v-2.67ZM16.5 9H22v6h-5.5l-3-3 3-3Zm-9 0H2v6h5.5l3-3-3-3ZM9 16.5l3-3 3 3V22H9v-5.5Z"
};
var speaker_group = {
  name: "speaker_group",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M9.8 1h8.4c.99 0 1.8.81 1.8 1.8v14.4c0 .99-.81 1.8-1.8 1.8l-8.4-.01c-.99 0-1.8-.8-1.8-1.79V2.8C8 1.81 8.81 1 9.8 1ZM18 17V3h-8v13.99l8 .01Zm-4-9a2 2 0 1 0 .001-3.999A2 2 0 0 0 14 8Zm3.5 4.5c0 1.93-1.57 3.5-3.5 3.5s-3.5-1.57-3.5-3.5S12.07 9 14 9s3.5 1.57 3.5 3.5ZM14 11c.83 0 1.5.67 1.5 1.5S14.83 14 14 14s-1.5-.67-1.5-1.5.67-1.5 1.5-1.5ZM4 5h2v16h10v2H6a2 2 0 0 1-2-2V5Z"
};
var speaker = {
  name: "speaker",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17 2H7c-1.1 0-2 .9-2 2v16c0 1.1.9 1.99 2 1.99L17 22c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2ZM7 20V4h10v16H7Zm7-13a2 2 0 1 1-4.001.001A2 2 0 0 1 14 7Zm-2 4c-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4-1.79-4-4-4Zm-2 4c0 1.1.9 2 2 2s2-.9 2-2-.9-2-2-2-2 .9-2 2Z"
};
var mouse = {
  name: "mouse",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 1.035c4.4 0 7.96 3.54 8 7.93v6c0 4.42-3.58 8-8 8s-8-3.58-8-8v-6c.04-4.39 3.6-7.93 8-7.93Zm1 7.93h5a6.005 6.005 0 0 0-5-5.84v5.84Zm-2-5.84v5.84H6a6.005 6.005 0 0 1 5-5.84Zm1 17.84c3.31 0 6-2.69 6-6v-4H6v4c0 3.31 2.69 6 6 6Z"
};
var keyboard_hide = {
  name: "keyboard_hide",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 2h16c1.1 0 2 .9 2 2v10c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2l.01-10c0-1.1.89-2 1.99-2Zm0 12h16V4H4v10Zm7-9h2v2h-2V5Zm2 3h-2v2h2V8ZM8 5h2v2H8V5Zm2 3H8v2h2V8ZM5 8h2v2H5V8Zm2-3H5v2h2V5Zm1 6h8v2H8v-2Zm8-3h-2v2h2V8Zm-2-3h2v2h-2V5Zm5 3h-2v2h2V8Zm-2-3h2v2h-2V5Zm-5 17-4-4h8l-4 4Z"
};
var keyboard = {
  name: "keyboard",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20 5H4c-1.1 0-1.99.9-1.99 2L2 17c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2Zm0 2v10H4V7h16Zm-7 1h-2v2h2V8Zm-2 3h2v2h-2v-2Zm-1-3H8v2h2V8Zm-2 3h2v2H8v-2Zm-1 0H5v2h2v-2ZM5 8h2v2H5V8Zm11 6H8v2h8v-2Zm-2-3h2v2h-2v-2Zm2-3h-2v2h2V8Zm1 3h2v2h-2v-2Zm2-3h-2v2h2V8Z"
};
var keyboard_voice = {
  name: "keyboard_voice",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 14.5c1.66 0 2.99-1.34 2.99-3l.01-6c0-1.66-1.34-3-3-3s-3 1.34-3 3v6c0 1.66 1.34 3 3 3Zm-1.2-9.1c0-.66.54-1.2 1.2-1.2.66 0 1.2.54 1.2 1.2l-.01 6.2c0 .66-.53 1.2-1.19 1.2-.66 0-1.2-.54-1.2-1.2V5.4ZM12 16.6c2.76 0 5.3-2.1 5.3-5.1H19c0 3.42-2.72 6.24-6 6.72v3.28h-2v-3.28c-3.28-.49-6-3.31-6-6.72h1.7c0 3 2.54 5.1 5.3 5.1Z"
};
var smartwatch = {
  name: "smartwatch",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16 0H8l-.95 5.73A7.94 7.94 0 0 0 4 12a7.94 7.94 0 0 0 3.05 6.27L8 24h8l.96-5.73A7.976 7.976 0 0 0 20 12c0-2.54-1.19-4.81-3.04-6.27L16 0Zm-1.28 4.48L14.31 2H9.7l-.41 2.47C10.13 4.17 11.05 4 12 4c.96 0 1.87.17 2.72.48ZM14.31 22l.41-2.48c-.85.31-1.76.48-2.72.48-.95 0-1.87-.17-2.71-.47L9.7 22h4.61ZM6 12c0 3.31 2.69 6 6 6s6-2.69 6-6-2.69-6-6-6-6 2.69-6 6Z"
};
var tv = {
  name: "tv",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 3h18c1.1 0 2 .9 2 2l-.01 12c0 1.1-.89 2-1.99 2h-5v2H8v-2H3c-1.1 0-2-.9-2-2V5c0-1.1.9-2 2-2Zm0 14h18V5H3v12Z"
};
var tablet_ipad = {
  name: "tablet_ipad",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 0H5a2.5 2.5 0 0 0-2.5 2.5v19A2.5 2.5 0 0 0 5 24h14a2.5 2.5 0 0 0 2.5-2.5v-19A2.5 2.5 0 0 0 19 0Zm-7 23c-.83 0-1.5-.67-1.5-1.5S11.17 20 12 20s1.5.67 1.5 1.5S12.83 23 12 23Zm-7.5-4h15V3h-15v16Z"
};
var tablet_android = {
  name: "tablet_android",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18 0H6C4.34 0 3 1.34 3 3v18c0 1.66 1.34 3 3 3h12c1.66 0 3-1.34 3-3V3c0-1.66-1.34-3-3-3Zm-4 22h-4v-1h4v1Zm-9.25-3h14.5V3H4.75v16Z"
};
var iphone = {
  name: "iphone",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16 1H8a2.5 2.5 0 0 0-2.5 2.5v17A2.5 2.5 0 0 0 8 23h8a2.5 2.5 0 0 0 2.5-2.5v-17A2.5 2.5 0 0 0 16 1Zm-4 21c-.83 0-1.5-.67-1.5-1.5S11.17 19 12 19s1.5.67 1.5 1.5S12.83 22 12 22Zm-4.5-4h9V4h-9v14Z"
};
var android = {
  name: "android",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16 1H8C6.34 1 5 2.34 5 4v16c0 1.66 1.34 3 3 3h8c1.66 0 3-1.34 3-3V4c0-1.66-1.34-3-3-3Zm1 17H7V4h10v14Zm-7 3h4v-1h-4v1Z"
};
var dock = {
  name: "dock",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16 1.01 8 1c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V3c0-1.1-.9-1.99-2-1.99ZM8 23h8v-2H8v2Zm0-8h8V5H8v10Z"
};
var device_unknown = {
  name: "device_unknown",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 1h10c1.1 0 2 .9 2 2v18c0 1.1-.9 2-2 2H7c-1.1 0-2-.9-2-2V3c0-1.1.9-2 2-2Zm10 18V5H7v14h10ZM12 6.72c-1.96 0-3.5 1.52-3.5 3.47h1.75c0-.93.82-1.75 1.75-1.75s1.75.82 1.75 1.75c0 .767-.505 1.163-1.072 1.608-.728.572-1.558 1.224-1.558 2.842h1.76c0-.945.61-1.488 1.24-2.05.678-.604 1.38-1.23 1.38-2.4 0-1.96-1.54-3.47-3.5-3.47ZM13 18v-2h-2v2h2Z"
};
var desktop_windows = {
  name: "desktop_windows",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 2h18c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2h-7v2h2v2H8v-2h2v-2H3c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2Zm0 14h18V4H3v12Z"
};
var desktop_mac = {
  name: "desktop_mac",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 2h18c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2h-7l2 3v1H8v-1l2-3H3c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2Zm0 12h18V4H3v10Z"
};
var computer = {
  name: "computer",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21.99 16c0 1.1-.89 2-1.99 2h4v2H0v-2h4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2h16c1.1 0 2 .9 2 2l-.01 10ZM20 6H4v10h16V6Z"
};
var google_cast_connected = {
  name: "google_cast_connected",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21 3H3c-1.1 0-2 .9-2 2v3h2V5h18v14h-7v2h7c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2ZM1 12v-2c6.07 0 11 4.92 11 11h-2a9 9 0 0 0-9-9Zm0 2v2c2.76 0 5 2.24 5 5h2c0-3.87-3.13-7-7-7Zm0 4v3h3c0-1.66-1.34-3-3-3Zm16-9H5V7h14v10h-5v-2h3V9Z"
};
var google_cast = {
  name: "google_cast",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 3h18c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2h-7v-2h7V5H3v3H1V5c0-1.1.9-2 2-2ZM1 21v-3c1.66 0 3 1.34 3 3H1Zm0-7v2c2.76 0 5 2.24 5 5h2c0-3.87-3.13-7-7-7Zm0-2v-2c6.07 0 11 4.92 11 11h-2a9 9 0 0 0-9-9Z"
};
var dns = {
  name: "dns",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20 3H4c-.55 0-1 .45-1 1v6c0 .55.45 1 1 1h16c.55 0 1-.45 1-1V4c0-.55-.45-1-1-1ZM5 9h14V5H5v4Zm-1 4h16c.55 0 1 .45 1 1v6c0 .55-.45 1-1 1H4c-.55 0-1-.45-1-1v-6c0-.55.45-1 1-1Zm15 2H5v4h14v-4ZM7 18.5c-.82 0-1.5-.67-1.5-1.5s.68-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5ZM5.5 7c0 .83.68 1.5 1.5 1.5.83 0 1.5-.68 1.5-1.5S7.82 5.5 7 5.5 5.5 6.17 5.5 7Z"
};
var fingerprint_scanner = {
  name: "fingerprint_scanner",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17.815 4.47c-.08 0-.16-.02-.23-.06-1.92-.99-3.58-1.41-5.57-1.41-1.98 0-3.86.47-5.57 1.41-.24.13-.54.04-.68-.2a.506.506 0 0 1 .2-.68C7.825 2.52 9.865 2 12.015 2c2.13 0 3.99.47 6.03 1.52.25.13.34.43.21.67a.49.49 0 0 1-.44.28ZM3.505 9.72a.499.499 0 0 1-.41-.79c.99-1.4 2.25-2.5 3.75-3.27 3.14-1.62 7.16-1.63 10.31-.01 1.5.77 2.76 1.86 3.75 3.25a.5.5 0 0 1-.12.7c-.23.16-.54.11-.7-.12a9.388 9.388 0 0 0-3.39-2.94c-2.87-1.47-6.54-1.47-9.4.01-1.36.7-2.5 1.7-3.4 2.96-.08.14-.23.21-.39.21Zm5.9 11.92c.09.1.22.15.35.15.13 0 .26-.05.37-.15.19-.2.19-.51 0-.71-.77-.78-1.21-1.27-1.85-2.42-.61-1.08-.93-2.41-.93-3.85 0-2.42 2.09-4.39 4.66-4.39 2.57 0 4.66 1.97 4.66 4.39 0 .28.22.5.5.5s.5-.22.5-.5c0-2.97-2.54-5.39-5.66-5.39s-5.66 2.42-5.66 5.39c0 1.61.36 3.11 1.05 4.34.67 1.21 1.14 1.77 2.01 2.64Zm7.52-1.7c-1.19 0-2.24-.3-3.1-.89-1.49-1.01-2.38-2.65-2.38-4.39 0-.28.22-.5.5-.5s.5.22.5.5c0 1.41.72 2.74 1.94 3.56.71.48 1.54.71 2.54.71.24 0 .64-.03 1.04-.1.27-.05.53.13.58.41.05.27-.13.53-.41.58-.57.11-1.07.12-1.21.12Zm-2.14 2.04c.04.01.09.02.13.02.21 0 .42-.15.47-.38a.496.496 0 0 0-.35-.61c-1.41-.39-2.32-.91-3.27-1.85a6.297 6.297 0 0 1-1.87-4.51c0-1.07.93-1.94 2.08-1.94s2.08.87 2.08 1.94c0 1.62 1.38 2.94 3.08 2.94 1.7 0 3.08-1.32 3.08-2.94 0-4.32-3.7-7.83-8.25-7.83-3.23 0-6.18 1.81-7.51 4.6-.45.95-.68 2.04-.68 3.24 0 1.35.24 2.65.73 3.96.09.25.38.39.64.29.26-.09.39-.38.29-.64-.6-1.6-.67-2.83-.67-3.61 0-1.04.2-1.99.59-2.8 1.17-2.45 3.77-4.03 6.61-4.03 4 0 7.25 3.06 7.25 6.83 0 1.07-.93 1.94-2.08 1.94s-2.08-.87-2.08-1.94c0-1.62-1.38-2.94-3.08-2.94-1.7 0-3.08 1.32-3.08 2.94 0 1.98.77 3.83 2.17 5.22 1.09 1.07 2.13 1.66 3.72 2.1Z"
};
var print_off = {
  name: "print_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2.23.82.82 2.23l5 4.99c-1.66 0-3 1.34-3 3v6h4v4h12l2.95 2.96 1.41-1.41L2.23.82Zm2.59 13.4v-4c0-.55.45-1 1-1h2l3 3h-4v2h-2Zm12 4-4-4h-4v4h8Zm-8-14h8v3h-5.34l2 2h6.34c.55 0 1 .45 1 1v4l-2 .01v-2.01h-2.34l4 4h2.34v-6c0-1.66-1.34-3-3-3h-1v-5h-12v.36l2 2v-.36Zm9 6.51a1 1 0 1 1 2 0 1 1 0 0 1-2 0Z"
};
var print = {
  name: "print",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 8h-1V3H6v5H5c-1.66 0-3 1.34-3 3v6h4v4h12v-4h4v-6c0-1.66-1.34-3-3-3ZM8 5h8v3H8V5Zm8 14v-4H8v4h8Zm2-4v-2H6v2H4v-4c0-.55.45-1 1-1h14c.55 0 1 .45 1 1v4h-2Zm-1-3.5a1 1 0 1 1 2 0 1 1 0 0 1-2 0Z"
};
var flash_off = {
  name: "flash_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18.43 2h-10v1.61l6.13 6.13L18.43 2Zm0 8h-3.61l2.28 2.28L18.43 10Zm-15-5.73 1.41-1.41 15.73 15.73L19.16 20l-4.15-4.15L11.43 22v-9h-3V9.27l-5-5Z"
};
var flash_on = {
  name: "flash_on",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 2v11h3v9l7-12h-4l3-8H7Z"
};
var style$1 = {
  name: "style",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m22.292 15.7-4.96-11.97a2.013 2.013 0 0 0-1.81-1.23c-.26 0-.53.04-.79.15L7.362 5.7a1.999 1.999 0 0 0-1.08 2.6l4.96 11.97a1.998 1.998 0 0 0 2.6 1.08l7.36-3.05a1.994 1.994 0 0 0 1.09-2.6Zm-19.5 3.7 1.34.56v-9.03l-2.43 5.86c-.41 1.02.08 2.19 1.09 2.61Zm5.34-11.86 4.96 11.96 7.35-3.05-4.95-11.95h-.01l-7.35 3.04Zm3.13.21a1 1 0 1 0 0 2 1 1 0 0 0 0-2ZM8.142 21.5c-1.1 0-2-.9-2-2v-6.34l3.45 8.34h-1.45Z"
};
var color_palette = {
  name: "color_palette",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 12c0 5.51 4.49 10 10 10a2.5 2.5 0 0 0 2.5-2.5c0-.61-.23-1.2-.64-1.67a.528.528 0 0 1-.13-.33c0-.28.22-.5.5-.5H16c3.31 0 6-2.69 6-6 0-4.96-4.49-9-10-9S2 6.49 2 12Zm2 0c0-4.41 3.59-8 8-8s8 3.14 8 7c0 2.21-1.79 4-4 4h-1.77a2.5 2.5 0 0 0-2.5 2.5c0 .6.22 1.19.63 1.65.06.07.14.19.14.35 0 .28-.22.5-.5.5-4.41 0-8-3.59-8-8Zm2.5-2a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3ZM8 7.5a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0ZM14.5 6a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm1.5 5.5a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0Z"
};
var dropper = {
  name: "dropper",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17.67 3c-.26 0-.51.1-.71.29l-3.12 3.12-1.93-1.91-1.41 1.41 1.42 1.42L3 16.25V21h4.75l8.92-8.92 1.42 1.42 1.41-1.41-1.92-1.92 3.12-3.12c.4-.4.4-1.03.01-1.42l-2.34-2.34c-.2-.19-.45-.29-.7-.29Zm-.01 2.41.92.92-2.69 2.69-.92-.92 2.69-2.69ZM5 17.08 6.92 19l8.06-8.06-1.92-1.92L5 17.08Z"
};
var texture = {
  name: "texture",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11.88 3 3 11.88v2.83L14.71 3h-2.83Zm7.63.08L3.08 19.51c.09.34.27.65.51.9.25.24.56.42.9.51L20.93 4.49c-.19-.69-.73-1.23-1.42-1.41ZM3 5c0-1.1.9-2 2-2h2L3 7V5Zm16 16c.55 0 1.05-.22 1.41-.59.37-.36.59-.86.59-1.41v-2l-4 4h2Zm-6.88 0H9.29L21 9.29v2.83L12.12 21Z"
};
var pram = {
  name: "pram",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M13.5 10V2c4.42 0 8 3.58 8 8h-8Zm2-5.66V8h3.66a6.032 6.032 0 0 0-3.66-3.66ZM6.94 11l-.95-2H2.5v2h2.22s1.89 4.07 2.12 4.42C5.74 16.01 5 17.17 5 18.5 5 20.43 6.57 22 8.5 22c1.76 0 3.22-1.3 3.46-3h2.08c.24 1.7 1.7 3 3.46 3 1.93 0 3.5-1.57 3.5-3.5 0-1.04-.46-1.97-1.18-2.61A7.948 7.948 0 0 0 21.5 11H6.94ZM7 18.5c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5S9.33 17 8.5 17 7 17.67 7 18.5ZM17.5 20c-.83 0-1.5-.67-1.5-1.5s.67-1.5 1.5-1.5 1.5.67 1.5 1.5-.67 1.5-1.5 1.5Zm.45-4.97.29-.37c.4-.51.71-1.07.92-1.66H7.87c.125.254.237.486.333.686.159.328.275.568.337.674l.44.67c1.18.17 2.18.93 2.68 1.97h2.68a3.505 3.505 0 0 1 3.61-1.97Z"
};
var fridge = {
  name: "fridge",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m6 2 12 .01c1.1 0 2 .88 2 1.99v16c0 1.1-.9 2-2 2H6c-1.1 0-2-.9-2-2V4a2 2 0 0 1 2-2Zm2 3h2v3H8V5Zm0 7h2v5H8v-5Zm10 8H6v-9.02h12V20ZM6 9h12V4H6v5Z"
};
var briefcase = {
  name: "briefcase",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20 7h-4V5l-2-2h-4L8 5v2H4c-1.1 0-2 .9-2 2v5c0 .75.4 1.38 1 1.73V19c0 1.11.89 2 2 2h14c1.11 0 2-.89 2-2v-3.28c.59-.35 1-.99 1-1.72V9c0-1.1-.9-2-2-2ZM10 5h4v2h-4V5Zm10 4H4v5h5v-3h6v3h5V9Zm-7 6h-2v-2h2v2Zm-8 4h14v-3h-4v1H9v-1H5v3Z"
};
var dice = {
  name: "dice",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2Zm0 2v14H5V5h14ZM6 16.5a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0ZM7.5 6a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm3 6a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0Zm6 3a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3ZM15 7.5a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0Z"
};
var toilet = {
  name: "toilet",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 6c1.11 0 2-.89 2-2 0-1.11-.89-2-2-2-1.11 0-2 .89-2 2 0 1.11.89 2 2 2Zm-2 8.5V22h4v-7.5h1.5V9c0-1.1-.9-2-2-2h-3c-1.1 0-2 .9-2 2v5.5H5ZM17.5 16v6h-3v-6h-3l2.54-7.63A2 2 0 0 1 15.94 7h.12c.86 0 1.62.55 1.9 1.37L20.5 16h-3ZM18 4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2Z"
};
var nature_people = {
  name: "nature_people",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M22.085 9.085c0-3.87-3.13-7-7-7s-7 3.13-7 7a6.98 6.98 0 0 0 5.83 6.89v3.94h-8v-3h1v-4c0-.55-.45-1-1-1h-3c-.55 0-1 .45-1 1v4h1v5h16v-2h-3v-3.88a7 7 0 0 0 6.17-6.95Zm-17.67-1.17a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm5.67 1.17c0 2.76 2.24 5 5 5s5-2.24 5-5-2.24-5-5-5-5 2.24-5 5Z"
};
var nature = {
  name: "nature",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12.885 16.035h.03v3.88h6v2h-14v-2h6v-3.94a6.98 6.98 0 0 1-5.83-6.89c0-3.87 3.13-7 7-7s7 3.13 7 7c0 3.59-2.71 6.55-6.2 6.95Zm-.8-11.95c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5Z"
};
var snow = {
  name: "snow",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M22 11h-4.17l3.24-3.24-1.41-1.42L15 11h-2V9l4.66-4.66-1.42-1.41L13 6.17V2h-2v4.17L7.76 2.93 6.34 4.34 11 9v2H9L4.34 6.34 2.93 7.76 6.17 11H2v2h4.17l-3.24 3.24 1.41 1.42L9 13h2v2l-4.66 4.66 1.42 1.41L11 17.83V22h2v-4.17l3.24 3.24 1.42-1.41L13 15v-2h2l4.66 4.66 1.41-1.42L17.83 13H22v-2Z"
};
var thermostat = {
  name: "thermostat",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2c1.66 0 3 1.34 3 3v8c1.21.91 2 2.37 2 4 0 2.76-2.24 5-5 5s-5-2.24-5-5c0-1.63.79-3.09 2-4V5c0-1.66 1.34-3 3-3Zm0 2c-.55 0-1 .45-1 1v6h2V9h-1V8h1V6h-1V5h1c0-.55-.45-1-1-1Z"
};
var money = {
  name: "money",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 4v16h20V4H2Zm5 4H5v8h2V8Zm5 8H9c-.55 0-1-.45-1-1V9c0-.55.45-1 1-1h3c.55 0 1 .45 1 1v6c0 .55-.45 1-1 1Zm3 0h3c.55 0 1-.45 1-1V9c0-.55-.45-1-1-1h-3c-.55 0-1 .45-1 1v6c0 .55.45 1 1 1Zm1-6h1v4h-1v-4Zm-6 0h1v4h-1v-4Zm-6 8h16V6H4v12Z"
};
var dollar = {
  name: "dollar",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20 4H4c-1.11 0-1.99.89-1.99 2L2 18c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V6c0-1.11-.89-2-2-2Zm-9 13h2v-1h1c.55 0 1-.45 1-1v-3c0-.55-.45-1-1-1h-3v-1h4V8h-2V7h-2v1h-1c-.55 0-1 .45-1 1v3c0 .55.45 1 1 1h3v1H9v2h2v1Zm-7 1h16V6H4v12Z"
};
var flower = {
  name: "flower",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M8.66 13.57c.15 0 .29-.01.43-.03A3.15 3.15 0 0 0 12 15.5c1.31 0 2.44-.81 2.91-1.96a3.145 3.145 0 0 0 3.57-3.11c0-.71-.25-1.39-.67-1.93.43-.54.67-1.22.67-1.93a3.145 3.145 0 0 0-3.57-3.11A3.15 3.15 0 0 0 12 1.5c-1.31 0-2.44.81-2.91 1.96a3.145 3.145 0 0 0-3.57 3.11c0 .71.25 1.39.67 1.93-.43.54-.68 1.22-.68 1.93 0 1.73 1.41 3.14 3.15 3.14ZM12 13.5c-.62 0-1.12-.49-1.14-1.1l.12-1.09c.32.12.66.19 1.02.19s.71-.07 1.03-.19l.11 1.09c-.02.61-.52 1.1-1.14 1.1Zm2.7-2.13c.18.13.4.2.64.2.63 0 1.15-.51 1.15-1.15 0-.44-.26-.84-.66-1.03l-.88-.42c-.12.74-.51 1.38-1.06 1.83l.81.57Zm-.01-5.74c.2-.13.42-.2.65-.2.63 0 1.14.51 1.14 1.14 0 .44-.25.83-.66 1.03l-.88.42c-.12-.74-.51-1.38-1.07-1.83l.82-.56ZM13.14 4.6c-.02-.61-.52-1.1-1.14-1.1-.62 0-1.12.49-1.14 1.1l.12 1.09c.32-.12.66-.19 1.02-.19s.71.07 1.03.19l.11-1.09Zm-4.48.83c.24 0 .46.07.64.2l.81.56c-.55.45-.94 1.09-1.06 1.83l-.88-.42c-.4-.2-.66-.59-.66-1.03 0-.63.52-1.14 1.15-1.14Zm.39 3.55-.88.42c-.4.2-.66.59-.65 1.02 0 .63.51 1.14 1.14 1.14.23 0 .45-.07.65-.2l.81-.55c-.56-.45-.95-1.09-1.07-1.83ZM12 22.5a9 9 0 0 0 9-9 9 9 0 0 0-9 9Zm-9-9a9 9 0 0 0 9 9 9 9 0 0 0-9-9Zm11.44 6.56c.71-1.9 2.22-3.42 4.12-4.12a7.04 7.04 0 0 1-4.12 4.12Zm-9-4.12c1.9.71 3.42 2.22 4.12 4.12a7.04 7.04 0 0 1-4.12-4.12Z"
};
var laundry = {
  name: "laundry",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m6 2 12 .01c1.11 0 2 .88 2 1.99v16c0 1.11-.89 2-2 2H6c-1.11 0-2-.89-2-2V4c0-1.11.89-2 2-2Zm-.01 2L6 20h12V4H5.99ZM8 5a1 1 0 1 0 0 2 1 1 0 0 0 0-2Zm2 1a1 1 0 1 1 2 0 1 1 0 0 1-2 0Zm2 13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5Zm2.36-2.64c1.3-1.3 1.3-3.42 0-4.72l-4.72 4.72c1.3 1.3 3.42 1.3 4.72 0Z"
};
var cake = {
  name: "cake",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14 5a2 2 0 1 1-4 0c0-.38.1-.73.29-1.03L12 1l1.71 2.97c.19.3.29.65.29 1.03Zm-1 5h5c1.66 0 3 1.34 3 3v9c0 .55-.45 1-1 1H4c-.55 0-1-.45-1-1v-9c0-1.66 1.34-3 3-3h5V8h2v2ZM5 21v-3c.9-.01 1.76-.37 2.4-1.01l1.09-1.07 1.07 1.07c1.31 1.31 3.59 1.3 4.89 0l1.08-1.07 1.07 1.07c.64.64 1.5 1 2.4 1.01v3H5Zm12.65-5.07c.36.37.84.56 1.35.57V13c0-.55-.45-1-1-1H6c-.55 0-1 .45-1 1v3.5c.51-.01.99-.21 1.34-.57l2.14-2.13 2.13 2.13c.74.74 2.03.74 2.77 0l2.14-2.13 2.13 2.13Z"
};
var cocktail = {
  name: "cocktail",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21 3H3v2l8 9v5H6v2h12v-2h-5v-5l8-9V3Zm-6.23 6L12 12.11 9.23 9h5.54ZM5.66 5l1.77 2h9.14l1.78-2H5.66Z"
};
var coffee = {
  name: "coffee",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 3h16c1.11 0 2 .89 2 2v3c0 1.11-.89 2-2 2h-2v3c0 2.21-1.79 4-4 4H8c-2.21 0-4-1.79-4-4V3Zm12 10V5H6v8c0 1.1.9 2 2 2h6c1.1 0 2-.9 2-2Zm2-5V5h2v3h-2ZM2 19h18v2H2v-2Z"
};
var drink = {
  name: "drink",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m3 2 2.01 18.23C5.13 21.23 5.97 22 7 22h10c1.03 0 1.87-.77 1.99-1.77L21 2H3Zm4 18.01L5.89 10H18.1L17 20l-10 .01ZM5.67 8h12.66l.43-4H5.23l.44 4ZM12 19c1.66 0 3-1.34 3-3 0-2-3-5.4-3-5.4S9 14 9 16c0 1.66 1.34 3 3 3Zm1-3c0-.36-.41-1.18-1-2.09-.59.9-1 1.72-1 2.09 0 .55.45 1 1 1s1-.45 1-1Z"
};
var pizza = {
  name: "pizza",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3.01 6C5.23 3.54 8.43 2 12 2c3.57 0 6.78 1.55 8.99 4L12 22 3.01 6Zm2.5.36L12 17.92l6.49-11.56A10.152 10.152 0 0 0 12 4c-2.38 0-4.68.85-6.49 2.36ZM9 5.5c-.83 0-1.5.67-1.5 1.5S8.17 8.5 9 8.5s1.5-.67 1.5-1.5S9.82 5.5 9 5.5Zm3 9c-.83 0-1.5-.67-1.5-1.5s.68-1.5 1.5-1.5 1.5.67 1.5 1.5-.68 1.5-1.5 1.5Z"
};
var fast_food = {
  name: "fast_food",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18 1v4h5l-1.65 16.53c-.1.82-.79 1.47-1.63 1.47H18v-2h1.39l1.4-14h-9.56L11 5h5V1h2ZM8.5 8.99C4.75 8.99 1 11 1 15h15c0-4-3.75-6.01-7.5-6.01ZM1 21.98c0 .56.45 1.01 1.01 1.01H15c.56 0 1.01-.45 1.01-1.01V21H1v.98Zm7.5-10.99c-1.41 0-3.77.46-4.88 2.01h9.76c-1.11-1.55-3.47-2.01-4.88-2.01ZM1 17h15v2H1v-2Z"
};
var restaurant = {
  name: "restaurant",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M9 9h2V2h2v7c0 2.21-1.79 4-4 4v9H7v-9c-2.21 0-4-1.79-4-4V2h2v7h2V2h2v7Zm7 5V6c0-1.76 2.24-4 5-4v20h-2v-8h-3Z"
};
var dining = {
  name: "dining",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19.85 9.939c-1.59 1.59-3.74 2.09-5.27 1.38l-1.47 1.47 6.88 6.88-1.41 1.41-6.88-6.88-6.89 6.87-1.41-1.41 9.76-9.76c-.71-1.53-.21-3.68 1.38-5.27 1.92-1.91 4.66-2.27 6.12-.81 1.47 1.47 1.1 4.21-.81 6.12Zm-9.22.36-2.83 2.83-4.19-4.18a4.008 4.008 0 0 1 0-5.66l7.02 7.01Z"
};
var puzzle = {
  name: "puzzle",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M8.25 4.75a2.5 2.5 0 0 1 5 0h4c1.1 0 2 .9 2 2v4a2.5 2.5 0 0 1 0 5v4c0 1.1-.9 2-2 2h-3.8v-.3c0-1.49-1.21-2.7-2.7-2.7-1.49 0-2.7 1.21-2.7 2.7v.3h-3.8c-1.1 0-2-.9-2-2v-3.8h.3c1.49 0 2.7-1.21 2.7-2.7 0-1.49-1.21-2.7-2.7-2.7h-.29v-3.8c0-1.1.89-2 1.99-2h4Zm3 0c0-.28-.22-.5-.5-.5s-.5.22-.5.5v2h-6l.01 2.12c1.75.68 2.99 2.39 2.99 4.38 0 1.99-1.25 3.7-3 4.38v2.12h2.12c.68-1.75 2.39-3 4.38-3 1.99 0 3.7 1.25 4.38 3h2.12v-6h2c.28 0 .5-.22.5-.5s-.22-.5-.5-.5h-2v-6h-6v-2Z"
};
var smoking_off = {
  name: "smoking_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16.03 8.175H14.5a3.35 3.35 0 0 1 0-6.7v1.5c-1.02 0-1.85.73-1.85 1.75s.83 2 1.85 2h1.53c1.87 0 3.47 1.35 3.47 3.16v1.64H18v-1.3c0-1.31-.92-2.05-1.97-2.05Zm.97 4.35h-2.34l2.34 2.34v-2.34Zm2.5 0H18v3h1.5v-3Zm2.5 0h-1.5v3H22v-3Zm-3.15-8.27c.62-.61 1-1.45 1-2.38h-1.5c0 1.02-.83 1.85-1.85 1.85v1.5c2.24 0 4 1.83 4 4.07v2.23H22v-2.24c0-2.22-1.28-4.14-3.15-5.03ZM2 5.525l1.41-1.41 17 17-1.41 1.41-7-7H2v-3h7l-7-7Z"
};
var smoking = {
  name: "smoking",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19.85 6.85c0 .93-.38 1.77-1 2.38 1.87.89 3.15 2.81 3.15 5.03v2.24h-1.5v-2.23c0-2.24-1.76-4.07-4-4.07V8.7c1.02 0 1.85-.83 1.85-1.85S17.52 5 16.5 5V3.5c1.85 0 3.35 1.5 3.35 3.35ZM14.5 11.7h1.53c1.87 0 3.47 1.35 3.47 3.16v1.64H18v-1.3c0-1.31-.92-2.05-1.97-2.05H14.5a3.35 3.35 0 0 1 0-6.7v1.5c-1.02 0-1.85.73-1.85 1.75s.83 2 1.85 2ZM2 17.5h15v3H2v-3Zm16 0h1.5v3H18v-3Zm2.5 0H22v3h-1.5v-3Z"
};
var widgets = {
  name: "widgets",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m16 2.345-5.66 5.65v-4.34h-8v8h8v-3.66l5.66 5.66h-3.66v8h8v-8H16l5.66-5.66L16 2.345Zm2.83 5.66L16 5.175l-2.83 2.83 2.83 2.83 2.83-2.83ZM8.34 9.655v-4h-4v4h4Zm10 6v4h-4v-4h4Zm-10 4v-4h-4v4h4Zm-6-6h8v8h-8v-8Z"
};
var puzzle_filled = {
  name: "puzzle_filled",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M10.75 2.25a2.5 2.5 0 0 0-2.5 2.5h-4c-1.1 0-1.99.9-1.99 2v3.8h.29c1.49 0 2.7 1.21 2.7 2.7 0 1.49-1.21 2.7-2.7 2.7h-.3v3.8c0 1.1.9 2 2 2h3.8v-.3c0-1.49 1.21-2.7 2.7-2.7 1.49 0 2.7 1.21 2.7 2.7v.3h3.8c1.1 0 2-.9 2-2v-4a2.5 2.5 0 0 0 0-5v-4c0-1.1-.9-2-2-2h-4a2.5 2.5 0 0 0-2.5-2.5Z"
};
var bandage = {
  name: "bandage",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m17.745 12 3.98-3.98a.996.996 0 0 0 0-1.41l-4.34-4.34a.996.996 0 0 0-1.41 0l-3.98 3.98-3.98-3.98a1.001 1.001 0 0 0-1.41 0l-4.34 4.34a.996.996 0 0 0 0 1.41L6.245 12l-3.98 3.98a.996.996 0 0 0 0 1.41l4.34 4.34c.39.39 1.02.39 1.41 0l3.98-3.98 3.98 3.98c.2.2.45.29.71.29.26 0 .51-.1.71-.29l4.34-4.34a.996.996 0 0 0 0-1.41L17.745 12Zm-5.73-3.02c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1Zm-8.34-1.66 3.63 3.62 3.62-3.63-3.62-3.62-3.63 3.63Zm6.34 5.66c-.55 0-1-.45-1-1s.45-1 1-1 1 .45 1 1-.45 1-1 1Zm1 1c0 .55.45 1 1 1s1-.45 1-1-.45-1-1-1-1 .45-1 1Zm3-3c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1Zm-.97 5.72 3.63 3.62 3.62-3.63-3.62-3.62-3.63 3.63Z"
};
var brush = {
  name: "brush",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19.169 3c-.26 0-.51.1-.71.29l-8.96 8.96 2.75 2.75 8.96-8.96a.996.996 0 0 0 0-1.41l-1.34-1.34c-.2-.2-.45-.29-.7-.29ZM7.499 16c.55 0 1 .45 1 1 0 1.1-.9 2-2 2-.17 0-.33-.02-.5-.05.31-.55.5-1.21.5-1.95 0-.55.45-1 1-1Zm-3 1c0-1.66 1.34-3 3-3s3 1.34 3 3c0 2.21-1.79 4-4 4-1.51 0-3.08-.78-4-2 .84 0 2-.69 2-2Z"
};
var measure = {
  name: "measure",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 6h18c1.1 0 2 .9 2 2v8c0 1.1-.9 2-2 2H3c-1.1 0-2-.9-2-2V8c0-1.1.9-2 2-2Zm0 10h18V8h-2v4h-2V8h-2v4h-2V8h-2v4H9V8H7v4H5V8H3v8Z"
};
var report_bug = {
  name: "report_bug",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17.19 8H20v2h-2.09c.05.33.09.66.09 1v1h2v2h-2v1c0 .34-.04.67-.09 1H20v2h-2.81c-1.04 1.79-2.97 3-5.19 3s-4.15-1.21-5.19-3H4v-2h2.09c-.05-.33-.09-.66-.09-1v-1H4v-2h2v-1c0-.34.04-.67.09-1H4V8h2.81c.45-.78 1.07-1.45 1.81-1.96L7 4.41 8.41 3l2.18 2.17c.45-.11.92-.17 1.41-.17.49 0 .96.06 1.42.17L15.59 3 17 4.41l-1.63 1.63c.75.51 1.37 1.18 1.82 1.96ZM16 15v-4c0-.22-.03-.47-.07-.69l-.1-.65-.38-.65c-.3-.53-.71-.97-1.21-1.31l-.61-.42-.68-.16a3.787 3.787 0 0 0-1.89 0l-.74.18-.57.39A4.1 4.1 0 0 0 8.54 9l-.37.65-.1.65c-.04.22-.07.47-.07.7v4c0 .23.03.48.07.71l.1.65.37.64c.72 1.23 2.04 2 3.46 2s2.74-.76 3.46-2l.37-.65.1-.65c.04-.23.07-.48.07-.7Zm-6-1h4v2h-4v-2Zm4-4h-4v2h4v-2Z"
};
var build_wrench = {
  name: "build_wrench",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m13.576 9.91 9.08 9.08c.4.4.4 1.03 0 1.41l-2.3 2.3a.996.996 0 0 1-1.41 0l-9.11-9.11c-2.32.87-5.03.38-6.89-1.48-2.3-2.29-2.51-5.88-.65-8.42l3.83 3.83 1.42-1.41-3.84-3.85c2.55-1.86 6.13-1.65 8.43.65a6.505 6.505 0 0 1 1.44 7Zm-3.38 1.22 9.46 9.46.88-.89-9.45-9.45c.46-.59.76-1.25.88-1.96.25-1.39-.16-2.88-1.24-3.96-.95-.94-2.2-1.38-3.44-1.31l3.09 3.09-4.24 4.24-3.09-3.09c-.07 1.24.37 2.5 1.32 3.44a4.472 4.472 0 0 0 3.83 1.25c.71-.1 1.39-.37 2-.82Z"
};
var gavel = {
  name: "gavel",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m18.375 6.66-2.83 2.83-5.66-5.66L12.715 1l5.66 5.66Zm-9.91-1.42-2.83 2.83 14.14 14.14 2.83-2.83L8.465 5.24ZM13.395 21h-12v2h12v-2Zm-3.51-5.86-5.66-5.66-2.83 2.83 5.66 5.66 2.83-2.83Z"
};
var placeholder_icon = {
  name: "placeholder_icon",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 5H3c0-1.1.9-2 2-2v2Zm0 8H3v-2h2v2Zm2 8h2v-2H7v2ZM5 9H3V7h2v2Zm8-6h-2v2h2V3Zm6 2V3c1.1 0 2 .9 2 2h-2ZM5 21v-2H3c0 1.1.9 2 2 2Zm0-4H3v-2h2v2ZM9 3H7v2h2V3Zm4 18h-2v-2h2v2Zm6-8h2v-2h-2v2Zm2 6c0 1.1-.9 2-2 2v-2h2ZM19 9h2V7h-2v2Zm2 8h-2v-2h2v2Zm-6 4h2v-2h-2v2Zm2-16h-2V3h2v2Z",
  sizes: {
    small: {
      name: "placeholder_icon_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M4 4H2c0-1.1.9-2 2-2v2Zm0 6H2V8h2v2Zm1 6h2v-2H5v2ZM4 7H2V5h2v2Zm6-5H8v2h2V2Zm4 2V2c1.1 0 2 .9 2 2h-2ZM4 16v-2H2c0 1.1.9 2 2 2Zm0-3H2v-2h2v2ZM7 2H5v2h2V2Zm3 14H8v-2h2v2Zm4-6h2V8h-2v2Zm2 4c0 1.1-.9 2-2 2v-2h2Zm-2-7h2V5h-2v2Zm2 6h-2v-2h2v2Zm-5 3h2v-2h-2v2Zm2-12h-2V2h2v2Z"
    }
  }
};
var offline_document = {
  name: "offline_document",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6ZM6 20V4h7v5h5v11H6Zm5-9h2v4h2.5L12 19l-3.5-4H11v-4Z"
};
var folder_shared = {
  name: "folder_shared",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 6h8c1.1 0 2 .9 2 2v10c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2l.01-12c0-1.1.89-2 1.99-2h6l2 2ZM4 6v12h16V8h-8.83l-2-2H4Zm11 7c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2Zm4 3v1h-8v-1c0-1.33 2.67-2 4-2s4 .67 4 2Z"
};
var folder_add = {
  name: "folder_add",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20 6h-8l-2-2H4c-1.11 0-1.99.89-1.99 2L2 18c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V8c0-1.11-.89-2-2-2ZM4 18V6h5.17l2 2H20v10H4Zm10-4h-2v-2h2v-2h2v2h2v2h-2v2h-2v-2Z"
};
var folder_open = {
  name: "folder_open",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 6h8c1.1 0 2 .9 2 2v10c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2l.01-12c0-1.1.89-2 1.99-2h6l2 2ZM4 8v10h16V8H4Z"
};
var folder = {
  name: "folder",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 4h6l2 2h8c1.1 0 2 .9 2 2v10c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2l.01-12c0-1.1.89-2 1.99-2Zm7.17 4-2-2H4v12h16V8h-8.83Z"
};
var file_description = {
  name: "file_description",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6Zm2 10H8v2h8v-2Zm0 4H8v2h8v-2ZM6 20h12V9h-5V4H6v16Z"
};
var file = {
  name: "file",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M6 2h8l6 6v12c0 1.1-.9 2-2 2H5.99C4.89 22 4 21.1 4 20l.01-16c0-1.1.89-2 1.99-2Zm0 2v16h12V9h-5V4H6Z"
};
var folder_favorite = {
  name: "folder_favorite",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20 6h-8l-2-2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2ZM4 18V6h5.17l2 2H20v10H4Zm8.39-1 .69-2.96-2.3-1.99 3.03-.26L15 9l1.19 2.79 3.03.26-2.3 1.99.69 2.96L15 15.47 12.39 17Z"
};
var file_add = {
  name: "file_add",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6Zm-1 9h-2v3H8v2h3v3h2v-3h3v-2h-3v-3Zm-7 9h12V9h-5V4H6v16Z"
};
var library_video = {
  name: "library_video",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M8 2h12c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H8c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2ZM2 6h2v14h14v2H4c-1.1 0-2-.9-2-2V6Zm18 10H8V4h12v12Zm-8-1.5v-9l6 4.5-6 4.5Z"
};
var library_add = {
  name: "library_add",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M8 2h12c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H8c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2ZM2 6h2v14h14v2H4c-1.1 0-2-.9-2-2V6Zm18 10H8V4h12v12Zm-5-2h-2v-3h-3V9h3V6h2v3h3v2h-3v3Z"
};
var library_books = {
  name: "library_books",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M8 2h12c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H8c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2ZM2 6h2v14h14v2H4c-1.1 0-2-.9-2-2V6Zm6 10V4h12v12H8Zm10-7h-8v2h8V9Zm-8 3h4v2h-4v-2Zm8-6h-8v2h8V6Z"
};
var library_music = {
  name: "library_music",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M8 2h12c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H8c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2Zm0 14h12V4H8v12Zm4.5-1a2.5 2.5 0 0 0 2.5-2.5V7h3V5h-4v5.51c-.42-.32-.93-.51-1.5-.51a2.5 2.5 0 0 0 0 5ZM2 6h2v14h14v2H4c-1.1 0-2-.9-2-2V6Z"
};
var library_image = {
  name: "library_image",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21 1H7c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V3c0-1.1-.9-2-2-2ZM1 5h2v16h16v2H3c-1.1 0-2-.9-2-2V5Zm12.21 8.83 2.75-3.54L19.5 15h-11l2.75-3.53 1.96 2.36ZM7 17h14V3H7v14Z"
};
var library_pdf = {
  name: "library_pdf",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M8 2h12c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H8c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2Zm0 14h12V4H8v12ZM4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6Zm12 3v3c0 .55-.45 1-1 1h-2V8h2c.55 0 1 .45 1 1Zm-2 0h1v3h-1V9Zm5 2h-1v2h-1V8h2v1h-1v1h1v1Zm-9 0h1c.55 0 1-.45 1-1V9c0-.55-.45-1-1-1H9v5h1v-2Zm1-2h-1v1h1V9Z"
};
var download = {
  name: "download",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 9.5h-4v-6H9v6H5l7 7 7-7Zm-8 2v-6h2v6h1.17L12 13.67 9.83 11.5H11Zm8 9v-2H5v2h14Z"
};
var upload = {
  name: "upload",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15 16.5v-6h4l-7-7-7 7h4v6h6ZM12 6.33l2.17 2.17H13v6h-2v-6H9.83L12 6.33Zm7 14.17v-2H5v2h14Z"
};
var download_done = {
  name: "download_done",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m5 10.7 4.6 4.6L19 6l-2-2-7.4 7.4L7 8.8l-2 1.9ZM19 18H5v2h14v-2Z"
};
var cloud_upload = {
  name: "cloud_upload",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19.35 10.04A7.49 7.49 0 0 0 12 4C9.11 4 6.6 5.64 5.35 8.04A5.994 5.994 0 0 0 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96ZM19 18H6c-2.21 0-4-1.79-4-4 0-2.05 1.53-3.76 3.56-3.97l1.07-.11.5-.95A5.469 5.469 0 0 1 12 6c2.62 0 4.88 1.86 5.39 4.43l.3 1.5 1.53.11A2.98 2.98 0 0 1 22 15c0 1.65-1.35 3-3 3Zm-8.45-5H8l4-4 4 4h-2.55v3h-2.9v-3Z"
};
var cloud_off = {
  name: "cloud_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4.41 2.93 3 4.34l2.77 2.77h-.42A5.994 5.994 0 0 0 0 13.07c0 3.31 2.69 6 6 6h11.73l2 2 1.41-1.41L4.41 2.93ZM24 14.07c0-2.64-2.05-4.78-4.65-4.96A7.49 7.49 0 0 0 12 3.07c-1.33 0-2.57.36-3.65.97l1.49 1.49c.67-.29 1.39-.46 2.16-.46 3.04 0 5.5 2.46 5.5 5.5v.5H19a2.996 2.996 0 0 1 1.79 5.4l1.41 1.41c1.09-.92 1.8-2.27 1.8-3.81Zm-22-1c0 2.21 1.79 4 4 4h9.73l-8-8H6c-2.21 0-4 1.79-4 4Z"
};
var cloud_download = {
  name: "cloud_download",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19.35 10.04A7.49 7.49 0 0 0 12 4C9.11 4 6.6 5.64 5.35 8.04A5.994 5.994 0 0 0 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96ZM19 18H6c-2.21 0-4-1.79-4-4 0-2.05 1.53-3.76 3.56-3.97l1.07-.11.5-.95A5.469 5.469 0 0 1 12 6c2.62 0 4.88 1.86 5.39 4.43l.3 1.5 1.53.11A2.98 2.98 0 0 1 22 15c0 1.65-1.35 3-3 3Zm-8.45-8h2.9v3H16l-4 4-4-4h2.55v-3Z"
};
var cloud_done = {
  name: "cloud_done",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19.35 10.04A7.49 7.49 0 0 0 12 4C9.11 4 6.6 5.64 5.35 8.04A5.994 5.994 0 0 0 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96ZM19 18H6c-2.21 0-4-1.79-4-4 0-2.05 1.53-3.76 3.56-3.97l1.07-.11.5-.95A5.469 5.469 0 0 1 12 6c2.62 0 4.88 1.86 5.39 4.43l.3 1.5 1.53.11A2.98 2.98 0 0 1 22 15c0 1.65-1.35 3-3 3ZM7.91 12.09 10 14.18l4.6-4.6 1.41 1.41L10 17l-3.5-3.5 1.41-1.41Z"
};
var cloud = {
  name: "cloud",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5.35 8.04A7.496 7.496 0 0 1 12 4a7.49 7.49 0 0 1 7.35 6.04c2.6.18 4.65 2.32 4.65 4.96 0 2.76-2.24 5-5 5H6c-3.31 0-6-2.69-6-6 0-3.09 2.34-5.64 5.35-5.96Zm12.04 2.39A5.503 5.503 0 0 0 12 6C9.94 6 8.08 7.14 7.13 8.97l-.5.95-1.07.11A3.973 3.973 0 0 0 2 14c0 2.21 1.79 4 4 4h13c1.65 0 3-1.35 3-3a2.98 2.98 0 0 0-2.78-2.96l-1.53-.11-.3-1.5Z"
};
var offline = {
  name: "offline",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2.02c-5.51 0-9.98 4.47-9.98 9.98 0 5.51 4.47 9.98 9.98 9.98 5.51 0 9.98-4.47 9.98-9.98 0-5.51-4.47-9.98-9.98-9.98Zm0 17.96c-4.4 0-7.98-3.58-7.98-7.98S7.6 4.02 12 4.02 19.98 7.6 19.98 12 16.4 19.98 12 19.98ZM8.25 13.5l4.5-8.5v5.5h3L11.39 19v-5.5H8.25Z"
};
var offline_saved = {
  name: "offline_saved",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 12C2 6.5 6.5 2 12 2s10 4.5 10 10-4.5 10-10 10S2 17.5 2 12Zm2 0c0 4.41 3.59 8 8 8s8-3.59 8-8-3.59-8-8-8-8 3.59-8 8Zm13 3v2H7v-2h10ZM8.4 9.3l1.9 1.9 5.3-5.3L17 7.3 10.3 14 7 10.7l1.4-1.4Z"
};
var collection_1 = {
  name: "collection_1",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21 1H7c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V3c0-1.1-.9-2-2-2ZM1 5h2v16h16v2H3c-1.1 0-2-.9-2-2V5Zm15 10h-2V7h-2V5h4v10Zm-9 2h14V3H7v14Z"
};
var collection_2 = {
  name: "collection_2",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 1h14c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H7c-1.1 0-2-.9-2-2V3c0-1.1.9-2 2-2ZM1 5h2v16h16v2H3c-1.1 0-2-.9-2-2V5Zm20 12H7V3h14v14Zm-8-4h4v2h-6v-4a2 2 0 0 1 2-2h2V7h-4V5h4a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2h-2v2Z"
};
var collection_3 = {
  name: "collection_3",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 1h14c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H7c-1.1 0-2-.9-2-2V3c0-1.1.9-2 2-2Zm0 16h14V3H7v14ZM3 5H1v16c0 1.1.9 2 2 2h16v-2H3V5Zm14 6.5V13a2 2 0 0 1-2 2h-4v-2h4v-2h-2V9h2V7h-4V5h4a2 2 0 0 1 2 2v1.5c0 .83-.67 1.5-1.5 1.5.83 0 1.5.67 1.5 1.5Z"
};
var collection_4 = {
  name: "collection_4",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21 1H7c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V3c0-1.1-.9-2-2-2ZM1 5h2v16h16v2H3c-1.1 0-2-.9-2-2V5Zm16 10h-2v-4h-4V5h2v4h2V5h2v10ZM7 17h14V3H7v14Z"
};
var collection_5 = {
  name: "collection_5",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 1h14c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H7c-1.1 0-2-.9-2-2V3c0-1.1.9-2 2-2Zm0 16h14V3H7v14ZM3 5H1v16c0 1.1.9 2 2 2h16v-2H3V5Zm14 6v2a2 2 0 0 1-2 2h-4v-2h4v-2h-4V5h6v2h-4v2h2a2 2 0 0 1 2 2Z"
};
var collection_6 = {
  name: "collection_6",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21 1H7c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V3c0-1.1-.9-2-2-2ZM3 5H1v16c0 1.1.9 2 2 2h16v-2H3V5Zm4 12h14V3H7v14Zm6-2h2a2 2 0 0 0 2-2v-2a2 2 0 0 0-2-2h-2V7h4V5h-4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2Zm2-4h-2v2h2v-2Z"
};
var collection_7 = {
  name: "collection_7",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 1h14c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H7c-1.1 0-2-.9-2-2V3c0-1.1.9-2 2-2ZM1 5h2v16h16v2H3c-1.1 0-2-.9-2-2V5Zm20 12H7V3h14v14ZM17 7l-4 8h-2l4-8h-4V5h6v2Z"
};
var collection_8 = {
  name: "collection_8",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 1h14c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H7c-1.1 0-2-.9-2-2V3c0-1.1.9-2 2-2ZM1 5h2v16h16v2H3c-1.1 0-2-.9-2-2V5Zm20 12H7V3h14v14Zm-6-2h-2a2 2 0 0 1-2-2v-1.5c0-.83.67-1.5 1.5-1.5-.83 0-1.5-.67-1.5-1.5V7a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v1.5c0 .83-.67 1.5-1.5 1.5.83 0 1.5.67 1.5 1.5V13a2 2 0 0 1-2 2Zm-2-8h2v2h-2V7Zm2 4h-2v2h2v-2Z"
};
var collection_9 = {
  name: "collection_9",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21 1H7c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V3c0-1.1-.9-2-2-2ZM3 5H1v16c0 1.1.9 2 2 2h16v-2H3V5Zm4 12h14V3H7v14Zm8-12h-2a2 2 0 0 0-2 2v2a2 2 0 0 0 2 2h2v2h-4v2h4a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2Zm-2 4h2V7h-2v2Z"
};
var collection_9_plus = {
  name: "collection_9_plus",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21 1H7c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V3c0-1.1-.9-2-2-2ZM3 5H1v16c0 1.1.9 2 2 2h16v-2H3V5Zm11 7V8a2 2 0 0 0-2-2h-1a2 2 0 0 0-2 2v1a2 2 0 0 0 2 2h1v1H9v2h3a2 2 0 0 0 2-2Zm-3-4v1h1V8h-1Zm8 1h2V3H7v14h14v-6h-2v2h-2v-2h-2V9h2V7h2v2Z"
};
var attachment = {
  name: "attachment",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18 16H6.5c-2.21 0-4-1.79-4-4s1.79-4 4-4H19a2.5 2.5 0 0 1 0 5H8.5c-.55 0-1-.45-1-1s.45-1 1-1H18V9.5H8.5a2.5 2.5 0 0 0 0 5H19c2.21 0 4-1.79 4-4s-1.79-4-4-4H6.5C3.46 6.5 1 8.96 1 12s2.46 5.5 5.5 5.5H18V16Z"
};
var attach_file = {
  name: "attach_file",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5a2.5 2.5 0 0 1 5 0v10.5c0 .55-.45 1-1 1s-1-.45-1-1V6H9.5v9.5a2.5 2.5 0 0 0 5 0V5c0-2.21-1.79-4-4-4s-4 1.79-4 4v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6H16Z"
};
var image = {
  name: "image",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2Zm0 2v14H5V5h14Zm-7.86 10.73 3-3.87L18 17H6l3-3.86 2.14 2.59Z"
};
var movie_file = {
  name: "movie_file",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m20 8-2-4h4v14c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2h1l2 4h3L8 4h2l2 4h3l-2-4h2l2 4h3Zm0 2H5.76L4 6.47V18h16v-8Z"
};
var slideshow = {
  name: "slideshow",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2Zm-9 13 5-4-5-4v8Zm-5 3h14V5H5v14Z"
};
var image_add = {
  name: "image_add",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19.5 1.5v3h3v2h-3v2.99s-1.99.01-2 0V6.5h-3s.01-1.99 0-2h3v-3h2Zm-2 19h-14v-14h9v-2h-9c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2v-9h-2v9Zm-7.79-3.17-1.96-2.36L5 18.5h11l-3.54-4.71-2.75 3.54Z"
};
var sun = {
  name: "sun",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11 1.05h2V4h-2V1.05Zm-6.04 2.5 1.8 1.79-1.42 1.41-1.79-1.79 1.41-1.41ZM4 11H1v2h3v-2Zm16.448-6.048L19.04 3.545l-1.788 1.789 1.407 1.407 1.789-1.79ZM17.24 18.66l1.79 1.8 1.41-1.41-1.8-1.79-1.4 1.4ZM23 11h-3v2h3v-2ZM12 6c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6-2.69-6-6-6Zm-4 6c0 2.21 1.79 4 4 4s4-1.79 4-4-1.79-4-4-4-4 1.79-4 4Zm5 8v2.95h-2V20h2Zm-8.04.45-1.41-1.41 1.79-1.8 1.41 1.41-1.79 1.8Z"
};
var battery_alert = {
  name: "battery_alert",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15.67 4H14V2h-4v2H8.33C7.6 4 7 4.6 7 5.33v15.33C7 21.4 7.6 22 8.33 22h7.33c.74 0 1.34-.6 1.34-1.33V5.33C17 4.6 16.4 4 15.67 4ZM13 18h-2v-2h2v2Zm-2-4h2V9h-2v5Z"
};
var battery_charging = {
  name: "battery_charging",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14 4h1.67C16.4 4 17 4.6 17 5.33v15.34c0 .73-.6 1.33-1.34 1.33H8.33C7.6 22 7 21.4 7 20.66V5.33C7 4.6 7.6 4 8.33 4H10V2h4v2Zm-3 10.5V20l4-7.5h-2V7l-4 7.5h2Z"
};
var battery = {
  name: "battery",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15.67 4H14V2h-4v2H8.33C7.6 4 7 4.6 7 5.33v15.33C7 21.4 7.6 22 8.33 22h7.33c.74 0 1.34-.6 1.34-1.33V5.33C17 4.6 16.4 4 15.67 4Z"
};
var battery_unknown = {
  name: "battery_unknown",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15.67 4H14V2h-4v2H8.33C7.6 4 7 4.6 7 5.33v15.33C7 21.4 7.6 22 8.33 22h7.33c.74 0 1.34-.6 1.34-1.33V5.33C17 4.6 16.4 4 15.67 4ZM13 16v2h-2v-2h2Zm.63-2.6c.29-.29.67-.71.67-.71.43-.43.7-1.03.7-1.69 0-1.66-1.34-3-3-3s-3 1.34-3 3h1.5c0-.83.67-1.5 1.5-1.5a1.498 1.498 0 0 1 1.06 2.56l-.93.94c-.47.48-.93 1.17-.93 2h1.6c0-.45.35-1.12.83-1.6Z"
};
var flame = {
  name: "flame",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M13.5 1.335s.74 2.65.74 4.8c0 2.06-1.35 3.73-3.41 3.73-2.07 0-3.63-1.67-3.63-3.73l.03-.36C5.21 8.175 4 11.285 4 14.665c0 4.42 3.58 8 8 8s8-3.58 8-8c0-5.39-2.59-10.2-6.5-13.33Zm-1.93 12.49c-1.36.28-2.17 1.16-2.17 2.41 0 1.34 1.11 2.42 2.49 2.42 2.05 0 3.71-1.66 3.71-3.71 0-1.07-.15-2.12-.46-3.12-.79 1.07-2.2 1.72-3.57 2Zm-5.57.84c0 3.31 2.69 6 6 6s6-2.69 6-6c0-2.56-.66-5.03-1.89-7.23-.53 2.6-2.62 4.43-5.28 4.43-1.56 0-2.96-.62-3.97-1.63-.56 1.39-.86 2.9-.86 4.43Z"
};
var waves = {
  name: "waves",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17 16.99c-1.35 0-2.2.42-2.95.8-.65.33-1.18.6-2.05.6-.9 0-1.4-.25-2.05-.6-.75-.38-1.57-.8-2.95-.8-1.38 0-2.2.42-2.95.8-.65.33-1.17.6-2.05.6v1.95c1.35 0 2.2-.42 2.95-.8.65-.33 1.17-.6 2.05-.6.88 0 1.4.25 2.05.6.75.38 1.57.8 2.95.8 1.38 0 2.2-.42 2.95-.8.65-.33 1.18-.6 2.05-.6.9 0 1.4.25 2.05.6.75.38 1.58.8 2.95.8v-1.95c-.9 0-1.4-.25-2.05-.6-.75-.38-1.6-.8-2.95-.8Zm0-4.45c-1.35 0-2.2.43-2.95.8-.65.32-1.18.6-2.05.6-.9 0-1.4-.25-2.05-.6-.75-.38-1.57-.8-2.95-.8-1.38 0-2.2.43-2.95.8-.65.32-1.17.6-2.05.6v1.95c1.35 0 2.2-.43 2.95-.8.65-.35 1.15-.6 2.05-.6.9 0 1.4.25 2.05.6.75.38 1.57.8 2.95.8 1.38 0 2.2-.43 2.95-.8.65-.35 1.15-.6 2.05-.6.9 0 1.4.25 2.05.6.75.38 1.58.8 2.95.8v-1.95c-.9 0-1.4-.25-2.05-.6-.75-.38-1.6-.8-2.95-.8Zm2.95-8.08c-.75-.38-1.58-.8-2.95-.8s-2.2.42-2.95.8c-.65.32-1.18.6-2.05.6-.9 0-1.4-.25-2.05-.6-.75-.37-1.57-.8-2.95-.8-1.38 0-2.2.42-2.95.8-.65.33-1.17.6-2.05.6v1.93c1.35 0 2.2-.43 2.95-.8.65-.33 1.17-.6 2.05-.6.88 0 1.4.25 2.05.6.75.38 1.57.8 2.95.8 1.38 0 2.2-.43 2.95-.8.65-.32 1.18-.6 2.05-.6.9 0 1.4.25 2.05.6.75.38 1.58.8 2.95.8V5.04c-.9 0-1.4-.25-2.05-.58ZM17 8.09c-1.35 0-2.2.43-2.95.8-.65.35-1.15.6-2.05.6-.9 0-1.4-.25-2.05-.6-.75-.38-1.57-.8-2.95-.8-1.38 0-2.2.43-2.95.8-.65.35-1.15.6-2.05.6v1.95c1.35 0 2.2-.43 2.95-.8.65-.32 1.18-.6 2.05-.6.87 0 1.4.25 2.05.6.75.38 1.57.8 2.95.8 1.38 0 2.2-.43 2.95-.8.65-.32 1.18-.6 2.05-.6.9 0 1.4.25 2.05.6.75.38 1.58.8 2.95.8V9.49c-.9 0-1.4-.25-2.05-.6-.75-.38-1.6-.8-2.95-.8Z"
};
var ev_station = {
  name: "ev_station",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m19.53 7.22-.01.01c.45.45.73 1.08.73 1.77v9.5a2.5 2.5 0 0 1-5 0v-5h-1.5V21h-10V5c0-1.1.9-2 2-2h6c1.1 0 2 .9 2 2v7h1c1.1 0 2 .9 2 2v4.5c0 .55.45 1 1 1s1-.45 1-1v-7.21c-.31.13-.64.21-1 .21a2.5 2.5 0 0 1-2.5-2.5c0-1.07.67-1.97 1.61-2.33l-2.11-2.11 1.06-1.06 3.72 3.72ZM11.75 19V5h-6v14h6Zm6-9c-.55 0-1-.45-1-1s.45-1 1-1 1 .45 1 1-.45 1-1 1Zm-12 3.5 4-7.5v5h2l-4 7v-4.5h-2Z"
};
var gas_station = {
  name: "gas_station",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m19.53 7.22-.01.01c.45.45.73 1.08.73 1.77v9.5a2.5 2.5 0 0 1-5 0v-5h-1.5V21h-10V5c0-1.1.9-2 2-2h6c1.1 0 2 .9 2 2v7h1c1.1 0 2 .9 2 2v4.5c0 .55.45 1 1 1s1-.45 1-1v-7.21c-.31.13-.64.21-1 .21a2.5 2.5 0 0 1-2.5-2.5c0-1.07.67-1.97 1.61-2.33l-2.11-2.11 1.06-1.06 3.72 3.72ZM11.75 19v-7h-6v7h6Zm0-9h-6V5h6v5Zm5-1c0 .55.45 1 1 1s1-.45 1-1-.45-1-1-1-1 .45-1 1Z"
};
var light = {
  name: "light",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M9 1.525h6v4.81c1.79 1.04 3 2.97 3 5.19 0 3.31-2.69 6-6 6s-6-2.69-6-6c0-2.22 1.21-4.15 3-5.19v-4.81Zm4 2v3.96l1 .58c1.24.72 2 2.04 2 3.46 0 2.21-1.79 4-4 4s-4-1.79-4-4c0-1.42.77-2.74 2-3.46l1-.58v-3.96h2Zm-9 6.95H1v2h3v-2Zm-.45 8.09 1.41 1.41 1.79-1.8-1.41-1.41-1.79 1.8Zm9.45.91v3h-2v-3h2Zm7-9h3v2h-3v-2Zm-.97 9.51-1.79-1.8 1.4-1.4 1.8 1.79-1.41 1.41Z"
};
var power_button = {
  name: "power_button",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14 3h2v4c1.1 0 2 .9 2 2v5.49L14.5 18v3h-5v-3L6 14.5V8.98C6 7.89 6.9 6.99 7.99 7H8V3h2v4h4V3Zm2 10.66V9H8v4.65l3.5 3.52V19h1v-1.83l3.5-3.51Z"
};
var power_button_off = {
  name: "power_button_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M8.215 2.505h2v3.88l-2-2v-1.88Zm8 9.88v-3.88h-3.88l-2-2h3.88v-4h2v4c1.1 0 2 .9 2 2v5.48l-.2.2-1.8-1.8Zm-11.88-9.04-1.41 1.41 3.29 3.29v5.96l3.5 3.5v3h5v-3l.48-.48 4.47 4.47 1.41-1.41-16.74-16.74Z"
};
var turbine = {
  name: "turbine",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 23h-1v-6.57C9.93 17.4 8.52 18 7 18c-3.25 0-6-2.75-6-6v-1h6.57C6.6 9.93 6 8.52 6 7c0-3.25 2.75-6 6-6h1v6.57C14.07 6.6 15.48 6 17 6c3.25 0 6 2.75 6 6v1h-6.57c.97 1.07 1.57 2.48 1.57 4 0 3.25-2.75 6-6 6Zm1-9.87v7.74c1.7-.46 3-2.04 3-3.87s-1.3-3.41-3-3.87ZM3.13 13c.46 1.7 2.04 3 3.87 3s3.41-1.3 3.87-3H3.13Zm10-2h7.74c-.46-1.7-2.05-3-3.87-3-1.82 0-3.41 1.3-3.87 3ZM11 3.13C9.3 3.59 8 5.18 8 7c0 1.82 1.3 3.41 3 3.87V3.13Z"
};
var lightbulb = {
  name: "lightbulb",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C8.14 2 5 5.14 5 9c0 2.38 1.19 4.47 3 5.74V17c0 .55.45 1 1 1h6c.55 0 1-.45 1-1v-2.26c1.81-1.27 3-3.36 3-5.74 0-3.86-3.14-7-7-7ZM9 21c0 .55.45 1 1 1h4c.55 0 1-.45 1-1v-1H9v1Zm5-7.3.85-.6A4.997 4.997 0 0 0 17 9c0-2.76-2.24-5-5-5S7 6.24 7 9c0 1.63.8 3.16 2.15 4.1l.85.6V16h4v-2.3Z"
};
var power = {
  name: "power",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11 3h2v10h-2V3Zm5.41 3.59 1.42-1.42A8.932 8.932 0 0 1 21 12a9 9 0 0 1-18 0c0-2.74 1.23-5.18 3.17-6.83l1.41 1.41A6.995 6.995 0 0 0 12 19c3.87 0 7-3.13 7-7a6.92 6.92 0 0 0-2.59-5.41Z"
};
var flare = {
  name: "flare",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11 1h2v6h-2V1ZM9.17 7.76 7.05 5.64 5.64 7.05l2.12 2.12 1.41-1.41ZM7 11H1v2h6v-2Zm11.36-3.95-1.41-1.41-2.12 2.12 1.41 1.41 2.12-2.12ZM17 11h6v2h-6v-2Zm-5-2c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3Zm4.95 9.36-2.12-2.12 1.41-1.41 2.12 2.12-1.41 1.41ZM5.64 16.95l1.41 1.41 2.12-2.12-1.41-1.41-2.12 2.12ZM13 23h-2v-6h2v6Z"
};
var electrical = {
  name: "electrical",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2 1 21h22L12 2Zm0 3.99L19.53 19H4.47L12 5.99Zm-1.397 8.836-1.357 3.558 4.342-3.9-1.246-.476L13.7 10.45l-4.342 3.901 1.246.475Z"
};
var wind_turbine = {
  name: "wind_turbine",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12.496 2.427A.496.496 0 0 0 12 2a.496.496 0 0 0-.496.427l-.888 6.929c-.27.259-.467.593-.56.968l-5.858 4.327a.48.48 0 0 0-.131.633.506.506 0 0 0 .627.205l6.272-2.581-.417 8.116H9V22h6v-.976h-1.55l-.414-8.084 6.267 2.588c.234.097.505.01.631-.203a.482.482 0 0 0-.135-.635l-5.842-4.304a1.997 1.997 0 0 0-.573-1.03l-.888-6.929Zm-.188 6.397L12 6.42l-.308 2.403a2.015 2.015 0 0 1 .616 0Zm-2.183 2.674-2.134 1.576 2.457-1.012a1.996 1.996 0 0 1-.323-.564Zm3.407.588 2.44 1.007-2.112-1.556c-.08.201-.191.386-.328.549ZM12 13.233l.446 7.791h-.892l.446-7.79Zm1-2.433a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"
};
var substation_onshore = {
  name: "substation_onshore",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 6a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2h-1v3h-2v-3H8v3H6v-3H5a2 2 0 0 1-2-2V6Zm16 0H5v6h14V6Zm-8.52 5.42.828-2.17-.76-.29 2.649-2.38-.828 2.17.76.29-2.65 2.38ZM21 18H3v2h18v-2Z"
};
var substation_offshore = {
  name: "substation_offshore",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 6a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2h-1v2l-2-1v-1H8v1l-2 1v-2H5a2 2 0 0 1-2-2V6Zm16 0H5v6h14V6Zm-2.998 10a6.985 6.985 0 0 1-4 1.28c-1.39 0-2.78-.43-4-1.28-1.22.85-2.61 1.32-4 1.32h-2v2h2c1.38 0 2.74-.35 4-.99 1.26.64 2.63.97 4 .97s2.74-.32 4-.97c1.26.65 2.62.99 4 .99h2v-2h-2c-1.39 0-2.78-.47-4-1.32Zm-4.694-6.75-.828 2.17 2.649-2.38-.76-.29.828-2.17-2.65 2.38.76.29Z"
};
var link_off = {
  name: "link_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5.11 7.445 2 4.335l1.41-1.41 16.74 16.74-1.41 1.41-4.01-4.01H13v-1.73l-2.27-2.27H8v-2h.73l-2.07-2.07a3.097 3.097 0 0 0-2.76 3.07c0 1.71 1.39 3.1 3.1 3.1h4v1.9H7c-2.76 0-5-2.24-5-5 0-2.09 1.29-3.88 3.11-4.62ZM17 7.065h-4v1.9h4c1.71 0 3.1 1.39 3.1 3.1 0 1.27-.77 2.37-1.87 2.84l1.4 1.4a4.986 4.986 0 0 0 2.37-4.24c0-2.76-2.24-5-5-5Zm-2.61 4 1.61 1.61v-1.61h-1.61Z"
};
var link = {
  name: "link",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11 15H7c-1.65 0-3-1.35-3-3s1.35-3 3-3h4V7H7c-2.76 0-5 2.24-5 5s2.24 5 5 5h4v-2Zm6-8h-4v2h4c1.65 0 3 1.35 3 3s-1.35 3-3 3h-4v2h4c2.76 0 5-2.24 5-5s-2.24-5-5-5Zm-1 4H8v2h8v-2Z"
};
var camera_add_photo = {
  name: "camera_add_photo",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M8.5 6.5h-3v3h-2v-3h-3v-2h3v-3h2v3h3v2Zm9.83 0h3.17c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2h-16c-1.1 0-2-.9-2-2v-9h2v9h16v-12h-4.05l-1.83-2H10.5v-2h6l1.83 2Zm-4.83 13c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5Zm0-8c1.65 0 3 1.35 3 3s-1.35 3-3 3-3-1.35-3-3 1.35-3 3-3Z"
};
var code = {
  name: "code",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m4.8 12 4.6 4.6L8 18l-6-6 6-6 1.4 1.4L4.8 12Zm14.4 0-4.6 4.6L16 18l6-6-6-6-1.4 1.4 4.6 4.6Z"
};
var gesture = {
  name: "gesture",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M6.38 5.68c-.31-.13-1.01.51-1.71 1.22L2.92 5.19c.22-.27.52-.57.88-.93.25-.25 1.4-1.25 2.72-1.25.87 0 2.51.69 2.51 2.86 0 1.36-.52 2.14-1.3 3.28-.45.66-1.5 2.43-1.85 3.52-.36 1.09-.09 1.92.36 1.92.412 0 .824-.496 1.078-.802l.032-.038c.23-.24 1.71-1.99 2.29-2.72.76-.93 2.69-2.84 4.94-2.84 2.94 0 3.88 2.55 4.03 4.2h2.47v2.5h-2.46c-.4 4.77-3.06 6.1-4.69 6.1-1.77 0-3.21-1.39-3.21-3.09s1.6-4.73 5.38-5.37l-.02-.154c-.1-.752-.215-1.636-1.74-1.636-1.25 0-2.87 1.95-4.08 3.44l-.013.015C9.143 15.558 8.266 16.641 7.2 16.95c-.9.27-1.89.1-2.64-.46-.86-.64-1.34-1.7-1.34-2.98 0-2.108 1.98-5.011 2.656-6.004.1-.147.172-.252.204-.306.3-.49.8-1.32.3-1.52Zm6.84 12.16c0 .46.43.72.74.72.7 0 1.83-.79 2.13-3.48-2.14.56-2.87 2.16-2.87 2.76Z"
};
var text_rotation_up = {
  name: "text_rotation_up",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m18 4-3 3h2v13h2V7h2l-3-3Zm-4 12.4-2.2-.9v-5l2.2-.9V7.5L3 12.25v1.5l11 4.75v-2.1Zm-4-5.27L4.98 13 10 14.87v-3.74Z"
};
var text_rotation_vertical = {
  name: "text_rotation_vertical",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m9.25 17-3 3-3-3h2V4h2v13h2ZM16 5h-1.5L9.75 16h2.1l.9-2.2h5l.9 2.2h2.1L16 5Zm-2.62 7 1.87-5.02L17.12 12h-3.74Z"
};
var text_rotation_angled_down = {
  name: "text_rotation_angled_down",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m11.875 8.23 3.54 3.54-.92 2.19 1.48 1.48 4.42-11.14-1.06-1.05-11.14 4.42 1.49 1.48 2.19-.92Zm3.75 12.52v-4.24l-1.41 1.41-9.2-9.19-1.41 1.41 9.19 9.19-1.41 1.42h4.24Zm.61-10.7 2.23-4.87-4.87 2.23 2.64 2.64Z"
};
var text_rotation_angled_up = {
  name: "text_rotation_angled_up",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m8.23 12.13 3.54-3.54 2.19.92 1.48-1.48L4.31 3.61 3.25 4.67l4.42 11.14 1.48-1.48-.92-2.2Zm8.28-3.75 1.41 1.41-9.19 9.19 1.41 1.41 9.19-9.19 1.42 1.42V8.38h-4.24ZM5.18 5.54l2.23 4.87 2.64-2.64-4.87-2.23Z"
};
var text_rotation_down = {
  name: "text_rotation_down",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m6 20 3-3H7V4H5v13H3l3 3Zm6.2-11.5v5l-2.2.9v2.1l11-4.75v-1.5L10 5.5v2.1l2.2.9Zm1.8 4.37L19.02 11 14 9.13v3.74Z"
};
var text_rotation_none = {
  name: "text_rotation_none",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m7.6 14 .9-2.2h5l.9 2.2h2.1L11.75 3h-1.5L5.5 14h2.1ZM20 18l-3-3v2H4v2h13v2l3-3Zm-7.13-8L11 4.98 9.13 10h3.74Z"
};
var opacity = {
  name: "opacity",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m12 2.345 5.66 5.65a8.02 8.02 0 0 1 2.34 5.64c0 2-.78 4.11-2.34 5.67a7.99 7.99 0 0 1-11.32 0C4.78 17.745 4 15.635 4 13.635s.78-4.08 2.34-5.64L12 2.345Zm-4.24 7.25c-1.14 1.13-1.75 2.4-1.76 4.4h12c-.01-2-.62-3.23-1.76-4.35L12 5.265l-4.24 4.33Z"
};
var invert_colors = {
  name: "invert_colors",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m12 2.345 5.66 5.66c3.12 3.12 3.12 8.19 0 11.31a7.98 7.98 0 0 1-5.66 2.34c-2.05 0-4.1-.78-5.66-2.34-3.12-3.12-3.12-8.19 0-11.31L12 2.345Zm-4.24 15.56a5.928 5.928 0 0 0 4.24 1.76V5.175l-4.24 4.25A5.928 5.928 0 0 0 6 13.665c0 1.6.62 3.1 1.76 4.24Z"
};
var flip_to_back = {
  name: "flip_to_back",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 5a2 2 0 0 1 2-2v2H7Zm2 2H7v2h2V7Zm0 4H7v2h2v-2Zm4 4h-2v2h2v-2Zm6-10V3c1.1 0 2 .9 2 2h-2Zm-6-2h-2v2h2V3ZM9 15v2a2 2 0 0 1-2-2h2Zm10-2h2v-2h-2v2Zm2-4h-2V7h2v2Zm-2 8c1.1 0 2-.9 2-2h-2v2ZM3 7h2v12h12v2H5a2 2 0 0 1-2-2V7Zm12-2h2V3h-2v2Zm2 12h-2v-2h2v2Z"
};
var flip_to_front = {
  name: "flip_to_front",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 3H9a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h10c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2ZM3 9h2V7H3v2Zm0 4h2v-2H3v2Zm0 4h2v-2H3v2Zm2 2v2a2 2 0 0 1-2-2h2Zm12 2h-2v-2h2v2Zm-8-6h10V5H9v10Zm2 6h2v-2h-2v2Zm-2 0H7v-2h2v2Z"
};
var insert_link = {
  name: "insert_link",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3.9 12c0-1.71 1.39-3.1 3.1-3.1h4V7H7c-2.76 0-5 2.24-5 5s2.24 5 5 5h4v-1.9H7c-1.71 0-3.1-1.39-3.1-3.1ZM8 13h8v-2H8v2Zm5-6h4c2.76 0 5 2.24 5 5s-2.24 5-5 5h-4v-1.9h4c1.71 0 3.1-1.39 3.1-3.1 0-1.71-1.39-3.1-3.1-3.1h-4V7Z"
};
var functions = {
  name: "functions",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18 4H6v2l6.5 6L6 18v2h12v-3h-7l5-5-5-5h7V4Z"
};
var format_strikethrough = {
  name: "format_strikethrough",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M6.85 8.575c0 .64.13 1.19.39 1.67.032.063.076.133.117.198l.033.052H12c-.64-.22-1.03-.45-1.41-.7-.49-.33-.74-.73-.74-1.21 0-.23.05-.45.15-.66.1-.21.25-.39.44-.55.19-.15.43-.27.72-.36.29-.09.64-.13 1.03-.13.4 0 .76.06 1.06.16.3.11.55.25.75.44.2.19.35.41.44.68.1.26.15.54.15.85h3.01c0-.66-.13-1.26-.38-1.81s-.61-1.03-1.08-1.43a4.94 4.94 0 0 0-1.69-.94c-.67-.23-1.4-.34-2.21-.34-.79 0-1.52.1-2.18.29-.65.2-1.22.48-1.7.83-.48.36-.85.79-1.11 1.29-.27.51-.4 1.06-.4 1.67ZM21 11.495v2.02h-3.87c.06.1.12.22.17.33.21.47.31 1.01.31 1.61 0 .64-.13 1.21-.38 1.71s-.61.93-1.07 1.27c-.46.34-1.02.6-1.67.79-.65.19-1.38.28-2.18.28-.48 0-.96-.05-1.44-.13-.48-.09-.94-.22-1.38-.39a5.69 5.69 0 0 1-1.22-.65c-.38-.26-.7-.57-.98-.92-.28-.36-.49-.76-.65-1.21-.16-.45-.24-1.03-.24-1.58h2.97c0 .45.11.9.25 1.21.14.31.34.56.59.75.25.19.56.33.91.42.35.09.75.13 1.18.13.38 0 .72-.05 1.01-.13.29-.09.52-.2.71-.35.19-.15.33-.33.42-.53.09-.21.14-.43.14-.66 0-.26-.04-.49-.11-.69-.08-.21-.22-.4-.43-.57-.21-.17-.5-.34-.87-.51a7.225 7.225 0 0 0-.269-.098c-.095-.033-.193-.067-.281-.102H3v-2h18Z"
};
var wrap_text = {
  name: "wrap_text",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19.5 4h-16v2h16V4Zm-16 14h6v-2h-6v2Zm0-8h13c2.21 0 4 1.79 4 4s-1.79 4-4 4h-2v2l-3-3 3-3v2h2.25c1.1 0 2-.9 2-2s-.9-2-2-2H3.5v-2Z"
};
var vertical_align_top = {
  name: "vertical_align_top",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 5V3h16v2H4Zm7 6H8l4-4 4 4h-3v10h-2V11Z"
};
var vertical_align_center = {
  name: "vertical_align_center",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16 5h-3V1h-2v4H8l4 4 4-4ZM8 19h3v4h2v-4h3l-4-4-4 4Zm-4-6v-2h16v2H4Z"
};
var vertical_align_bottom = {
  name: "vertical_align_bottom",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M13 13h3l-4 4-4-4h3V3h2v10Zm-9 8v-2h16v2H4Z"
};
var title = {
  name: "title",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 4.5v3h5.5v12h3v-12H19v-3H5Z"
};
var text_field = {
  name: "text_field",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2.5 7.5v-3h13v3h-5v12h-3v-12h-5Zm10 2h9v3h-3v7h-3v-7h-3v-3Z"
};
var format_shape = {
  name: "format_shape",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M23 1v6h-2v10h2v6h-6v-2H7v2H1v-6h2V7H1V1h6v2h10V1h6ZM5 3H3v2h2V3Zm0 18H3v-2h2v2Zm2-4v2h10v-2h2V7h-2V5H7v2H5v10h2Zm14 4h-2v-2h2v2ZM19 3v2h2V3h-2Zm-5.27 11h-3.49l-.73 2H7.89l3.4-9h1.4l3.41 9h-1.63l-.74-2Zm-.43-1.26h-2.61L12 8.91l1.3 3.83Z"
};
var format_quote = {
  name: "format_quote",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M8.62 18H3.38l2-4H3V6h8v7.24L8.62 18Zm4.76 0h5.24L21 13.24V6h-8v8h2.38l-2 4Zm4-2h-.76l2-4H15V8h4v4.76L17.38 16Zm-10 0h-.76l2-4H5V8h4v4.76L7.38 16Z"
};
var format_list_numbered = {
  name: "format_list_numbered",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4.5 8h-1V5h-1V4h2v4Zm0 9.5V17h-2v-1h3v4h-3v-1h2v-.5h-1v-1h1Zm-2-6.5h1.8l-1.8 2.1v.9h3v-1H3.7l1.8-2.1V10h-3v1Zm5-4V5h14v2h-14Zm0 12h14v-2h-14v2Zm14-6h-14v-2h14v2Z"
};
var format_list_bulleted = {
  name: "format_list_bulleted",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2.75 6c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5-.67 1.5-1.5 1.5-1.5-.67-1.5-1.5Zm0 6c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5-.67 1.5-1.5 1.5-1.5-.67-1.5-1.5Zm1.5 4.5c-.83 0-1.5.68-1.5 1.5s.68 1.5 1.5 1.5 1.5-.68 1.5-1.5-.67-1.5-1.5-1.5Zm17 2.5h-14v-2h14v2Zm-14-6h14v-2h-14v2Zm0-6V5h14v2h-14Z"
};
var format_line_spacing = {
  name: "format_line_spacing",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M8.75 7h-2.5v10h2.5l-3.5 3.5-3.5-3.5h2.5V7h-2.5l3.5-3.5L8.75 7Zm1.5 0V5h12v2h-12Zm0 12h12v-2h-12v2Zm12-6h-12v-2h12v2Z"
};
var format_size = {
  name: "format_size",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M8.5 7.5v-3h13v3h-5v12h-3v-12h-5Zm-3 5h-3v-3h9v3h-3v7h-3v-7Z"
};
var format_highlight = {
  name: "format_highlight",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M13.001 2h-2v3h2V2Zm-7 12 3 3v5h6v-5l3-3V9h-12v5Zm2-3h8v2.17l-3 3V20h-2v-3.83l-3-3V11ZM3.503 5.874 4.917 4.46l2.122 2.12-1.414 1.416-2.122-2.12Zm15.58-1.411-2.122 2.12 1.413 1.415 2.123-2.12-1.413-1.415Z"
};
var format_underline = {
  name: "format_underline",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18 11c0 3.31-2.69 6-6 6s-6-2.69-6-6V3h2.5v8c0 1.93 1.57 3.5 3.5 3.5s3.5-1.57 3.5-3.5V3H18v8ZM5 21v-2h14v2H5Z"
};
var format_italics = {
  name: "format_italics",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M10 5v3h2.21l-3.42 8H6v3h8v-3h-2.21l3.42-8H18V5h-8Z"
};
var format_indent_increase = {
  name: "format_indent_increase",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 5V3h18v2H3Zm4 7-4 4V8l4 4Zm14 9H3v-2h18v2Zm-10-4h10v-2H11v2Zm0-8h10V7H11v2Zm10 4H11v-2h10v2Z"
};
var format_indent_decrease = {
  name: "format_indent_decrease",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 5V3h18v2H3Zm4 11-4-4 4-4v8Zm14 1H11v-2h10v2ZM3 21h18v-2H3v2Zm8-12h10V7H11v2Zm10 4H11v-2h10v2Z"
};
var format_color_text = {
  name: "format_color_text",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m11 1.5-5.5 14h2.25l1.12-3h6.25l1.12 3h2.25L13 1.5h-2Zm-1.38 9L12 4.17l2.38 6.33H9.62Zm14.38 8H0v4h24v-4Z"
};
var format_color_reset = {
  name: "format_color_reset",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 6.195c1.53 2 3.08 4.43 3.71 6.24l2.23 2.23c.03-.27.06-.55.06-.83 0-3.98-6-10.8-6-10.8s-1.18 1.35-2.5 3.19l1.44 1.44c.34-.51.7-1 1.06-1.47Zm-6.59-1.22L4 6.385l3.32 3.32c-.77 1.46-1.32 2.92-1.32 4.13 0 3.31 2.69 6 6 6 1.52 0 2.9-.57 3.95-1.5l2.63 2.63 1.42-1.41L5.41 4.975ZM8 13.835c0 2.21 1.79 4 4 4 .96 0 1.83-.36 2.53-.92l-5.72-5.72c-.49 1.02-.81 1.95-.81 2.64Z"
};
var format_color_fill = {
  name: "format_color_fill",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m7.62 0 8.94 8.94c.59.59.59 1.54 0 2.12l-5.5 5.5c-.29.29-.68.44-1.06.44s-.77-.15-1.06-.44l-5.5-5.5a1.49 1.49 0 0 1 0-2.12l5.15-5.15-2.38-2.38L7.62 0ZM10 5.21 5.21 10h9.58L10 5.21Zm9 6.29s-2 2.17-2 3.5c0 1.1.9 2 2 2s2-.9 2-2c0-1.33-2-3.5-2-3.5Zm5 8.5H0v4h24v-4Z"
};
var format_clear = {
  name: "format_clear",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m3 5.34 1.41-1.41 14.73 14.73-1.41 1.41-5.66-5.66-1.57 3.66h-3l2.47-5.76L3 5.34Zm18-1.27v3h-5.79l-1.45 3.38-2.09-2.1.55-1.28h-1.83l-3-3H21Z"
};
var format_bold = {
  name: "format_bold",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15.225 11.79c.97-.67 1.65-1.77 1.65-2.79 0-2.26-1.75-4-4-4h-6.25v14h7.04c2.09 0 3.71-1.7 3.71-3.79 0-1.52-.86-2.82-2.15-3.42Zm-5.6-4.29h3c.83 0 1.5.67 1.5 1.5s-.67 1.5-1.5 1.5h-3v-3Zm0 9h3.5c.83 0 1.5-.67 1.5-1.5s-.67-1.5-1.5-1.5h-3.5v3Z"
};
var format_align_right = {
  name: "format_align_right",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 5V3h18v2H3Zm6 4h12V7H9v2Zm12 4H3v-2h18v2ZM9 17h12v-2H9v2Zm-6 4h18v-2H3v2Z"
};
var format_align_left = {
  name: "format_align_left",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 5V3h18v2H3Zm12 2H3v2h12V7Zm0 8H3v2h12v-2Zm6-2H3v-2h18v2ZM3 21h18v-2H3v2Z"
};
var format_align_justify = {
  name: "format_align_justify",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 5V3h18v2H3Zm0 4h18V7H3v2Zm18 4H3v-2h18v2ZM3 17h18v-2H3v2Zm0 4h18v-2H3v2Z"
};
var format_align_center = {
  name: "format_align_center",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 5V3h18v2H3Zm4 2v2h10V7H7Zm14 6H3v-2h18v2ZM7 15v2h10v-2H7Zm-4 6h18v-2H3v2Z"
};
var drag_handle = {
  name: "drag_handle",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 9h16v2H4V9Zm16 6H4v-2h16v2Z"
};
var keyboard_space_bar = {
  name: "keyboard_space_bar",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M18 9v4H6V9H4v6h16V9h-2Z"
};
var border_vertical = {
  name: "border_vertical",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 5h2V3H3v2Zm0 4h2V7H3v2Zm6 12H7v-2h2v2Zm-2-8h2v-2H7v2Zm-2 0H3v-2h2v2Zm-2 8h2v-2H3v2Zm2-4H3v-2h2v2ZM7 5h2V3H7v2Zm14 12h-2v-2h2v2Zm-10 4h2V3h-2v18Zm10 0h-2v-2h2v2Zm-2-8h2v-2h-2v2Zm0-8V3h2v2h-2Zm0 4h2V7h-2v2Zm-2-4h-2V3h2v2Zm-2 16h2v-2h-2v2Zm2-8h-2v-2h2v2Z"
};
var border_top = {
  name: "border_top",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 3v2h18V3H3Zm0 6h2V7H3v2Zm4 4h2v-2H7v2Zm0 8h2v-2H7v2Zm6-8h-2v-2h2v2Zm-2 8h2v-2h-2v2Zm-6-4H3v-2h2v2Zm-2 4h2v-2H3v2Zm2-8H3v-2h2v2Zm8 4h-2v-2h2v2Zm6-8h2V7h-2v2Zm2 4h-2v-2h2v2Zm0 4h-2v-2h2v2Zm-6 4h2v-2h-2v2ZM13 9h-2V7h2v2Zm6 12h2v-2h-2v2Zm-2-8h-2v-2h2v2Z"
};
var border_style = {
  name: "border_style",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 3v18h2V5h16V3H3Zm18 10h-2v-2h2v2Zm-2 4h2v-2h-2v2ZM7 21h2v-2H7v2Zm10 0h-2v-2h2v2Zm4 0h-2v-2h2v2Zm-8 0h-2v-2h2v2Zm8-12h-2V7h2v2Z"
};
var border_right = {
  name: "border_right",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 5h2V3H3v2Zm4 16h2v-2H7v2ZM9 5H7V3h2v2Zm-2 8h2v-2H7v2Zm-2 8H3v-2h2v2Zm6 0h2v-2h-2v2Zm-6-8H3v-2h2v2Zm-2 4h2v-2H3v2Zm2-8H3V7h2v2Zm6 8h2v-2h-2v2Zm6-4h-2v-2h2v2Zm2-10v18h2V3h-2Zm-2 18h-2v-2h2v2ZM15 5h2V3h-2v2Zm-2 8h-2v-2h2v2Zm-2-8h2V3h-2v2Zm2 4h-2V7h2v2Z"
};
var border_outer = {
  name: "border_outer",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 3v18h18V3H3Zm10 4h-2v2h2V7Zm0 4h-2v2h2v-2Zm2 0h2v2h-2v-2ZM5 19h14V5H5v14Zm8-4h-2v2h2v-2Zm-6-4h2v2H7v-2Z"
};
var border_left = {
  name: "border_left",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 21H3V3h2v18Zm8-16h-2V3h2v2Zm-2 12h2v-2h-2v2Zm0 4h2v-2h-2v2Zm0-12h2V7h-2v2Zm2 4h-2v-2h2v2Zm-6 8h2v-2H7v2ZM9 5H7V3h2v2Zm-2 8h2v-2H7v2Zm12-4h2V7h-2v2Zm-2 12h-2v-2h2v2Zm2-4h2v-2h-2v2Zm0-12V3h2v2h-2Zm0 8h2v-2h-2v2Zm2 8h-2v-2h2v2Zm-6-8h2v-2h-2v2Zm2-8h-2V3h2v2Z"
};
var border_inner = {
  name: "border_inner",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 3h2v2H7V3ZM3 7h2v2H3V7Zm0 14h2v-2H3v2Zm4 0h2v-2H7v2Zm-4-4h2v-2H3v2ZM5 3H3v2h2V3Zm10 0h2v2h-2V3Zm4 6h2V7h-2v2Zm0-4V3h2v2h-2Zm-4 16h2v-2h-2v2ZM11 3h2v8h8v2h-8v8h-2v-8H3v-2h8V3Zm8 18h2v-2h-2v2Zm2-4h-2v-2h2v2Z"
};
var border_horizontal = {
  name: "border_horizontal",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 3h2v2H3V3Zm2 4H3v2h2V7ZM3 21h2v-2H3v2Zm2-4H3v-2h2v2Zm2 4h2v-2H7v2ZM9 3H7v2h2V3Zm6 0h2v2h-2V3Zm-2 4h-2v2h2V7Zm-2-4h2v2h-2V3Zm8 14h2v-2h-2v2Zm-6 4h-2v-2h2v2ZM3 13h18v-2H3v2Zm16-8V3h2v2h-2Zm0 4h2V7h-2v2Zm-6 8h-2v-2h2v2Zm2 4h2v-2h-2v2Zm6 0h-2v-2h2v2Z"
};
var border_color = {
  name: "border_color",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20.71 4.042a.996.996 0 0 0 0-1.41L18.37.292a.996.996 0 0 0-1.41 0L15 2.252l3.75 3.75 1.96-1.96ZM4 13.252l10-10 3.75 3.75L7.75 17H4v-3.75ZM6 15h.92l8-8-.92-.92-8 8v.92Zm18 5H0V24h24v-4Z"
};
var border_clear = {
  name: "border_clear",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 5h2V3H7v2Zm0 8h2v-2H7v2Zm2 8H7v-2h2v2Zm2-4h2v-2h-2v2Zm2 4h-2v-2h2v2ZM3 21h2v-2H3v2Zm2-4H3v-2h2v2Zm-2-4h2v-2H3v2Zm2-4H3V7h2v2ZM3 5h2V3H3v2Zm10 8h-2v-2h2v2Zm6 4h2v-2h-2v2Zm2-4h-2v-2h2v2Zm-2 8h2v-2h-2v2Zm2-12h-2V7h2v2ZM11 9h2V7h-2v2Zm8-4V3h2v2h-2Zm-8 0h2V3h-2v2Zm6 16h-2v-2h2v2Zm-2-8h2v-2h-2v2Zm2-8h-2V3h2v2Z"
};
var border_bottom = {
  name: "border_bottom",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 3h2v2H3V3Zm4 0h2v2H7V3Zm2 8H7v2h2v-2Zm4 4h-2v2h2v-2Zm0-4h-2v2h2v-2Zm0-4h-2v2h2V7Zm2 4h2v2h-2v-2Zm-2-8h-2v2h2V3Zm2 0h2v2h-2V3Zm4 10h2v-2h-2v2Zm2 4h-2v-2h2v2ZM5 7H3v2h2V7Zm14-2V3h2v2h-2Zm0 4h2V7h-2v2ZM3 11h2v2H3v-2Zm0 10h18v-2H3v2Zm0-6h2v2H3v-2Z"
};
var border_all = {
  name: "border_all",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M3 3v18h18V3H3Zm8 16H5v-6h6v6Zm-6-8h6V5H5v6Zm14 8h-6v-6h6v6Zm-6-8h6V5h-6v6Z"
};
var keyboard_backspace = {
  name: "keyboard_backspace",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21 11H6.83l3.58-3.59L9 6l-6 6 6 6 1.41-1.41L6.83 13H21v-2Z"
};
var keyboard_capslock = {
  name: "keyboard_capslock",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16.59 13.205 12 8.615l-4.59 4.59L6 11.795l6-6 6 6-1.41 1.41Zm1.41 3v2H6v-2h12Z"
};
var keyboard_return = {
  name: "keyboard_return",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19.5 7v4H6.33l3.58-3.59L8.5 6l-6 6 6 6 1.41-1.41L6.33 13H21.5V7h-2Z"
};
var keyboard_tab = {
  name: "keyboard_tab",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m15.67 11-3.58-3.59L13.5 6l6 6-6 6-1.42-1.41L15.67 13H1.5v-2h14.17Zm6.83 7h-2V6h2v12Z"
};
var rotate_3d = {
  name: "rotate_3d",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m11.345.03.66-.03c6.29 0 11.44 4.84 11.94 10.99h-1.5c-.35-3.76-2.69-6.93-5.96-8.48l-1.33 1.33-3.81-3.81Zm-2.93 14.93c-.19 0-.37-.03-.52-.08a1.07 1.07 0 0 1-.4-.24c-.11-.1-.2-.22-.26-.37-.06-.14-.09-.3-.09-.47h-1.3c0 .36.07.68.21.95.14.27.33.5.56.69.24.18.51.32.82.41.3.1.62.15.96.15.37 0 .72-.05 1.03-.15.32-.1.6-.25.83-.44.23-.19.42-.43.55-.72.13-.29.2-.61.2-.97 0-.19-.02-.38-.07-.56a1.67 1.67 0 0 0-.23-.51c-.1-.16-.24-.3-.4-.43-.17-.13-.37-.23-.61-.31a2.097 2.097 0 0 0 .89-.75c.1-.15.17-.3.22-.46.05-.16.07-.32.07-.48 0-.36-.06-.68-.18-.96a1.78 1.78 0 0 0-.51-.69c-.2-.19-.47-.33-.77-.43-.31-.09-.65-.14-1.02-.14-.36 0-.69.05-1 .16-.3.11-.57.26-.79.45-.21.19-.38.41-.51.67-.12.26-.18.54-.18.85h1.3c0-.17.03-.32.09-.45a.94.94 0 0 1 .25-.34c.11-.09.23-.17.38-.22.15-.05.3-.08.48-.08.4 0 .7.1.89.31.19.2.29.49.29.86 0 .18-.03.34-.08.49a.87.87 0 0 1-.25.37c-.11.1-.25.18-.41.24-.16.06-.36.09-.58.09h-.77v1.03h.77c.22 0 .42.02.6.07s.33.13.45.23c.12.11.22.24.29.4.07.16.1.35.1.57 0 .41-.12.72-.35.93-.23.23-.55.33-.95.33Zm-.89 6.52A10.487 10.487 0 0 1 1.555 13h-1.5c.51 6.16 5.66 11 11.95 11l.66-.03-3.81-3.81-1.33 1.32Zm8.3-13.21c.44.18.82.44 1.14.77.32.33.57.73.74 1.2.17.47.26.99.26 1.57v.4c0 .58-.09 1.1-.26 1.57-.17.46-.42.86-.74 1.19-.32.33-.71.58-1.16.76-.45.18-.96.27-1.51.27h-2.3V8h2.36c.54 0 1.03.09 1.47.27Zm.75 3.93c0 .42-.05.79-.14 1.13-.1.33-.24.62-.43.85-.19.23-.43.41-.71.53-.29.12-.62.18-.99.18h-.91V9.12h.97c.72 0 1.27.23 1.64.69.38.46.57 1.12.57 1.99v.4Z"
};
var spellcheck = {
  name: "spellcheck",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11.72 15.25h2.09l-5.11-13H6.84l-5.11 13h2.09l1.12-3h5.64l1.14 3Zm-6.02-5 2.07-5.52 2.07 5.52H5.7Zm7.07 8.68 8.09-8.09 1.41 1.41-9.49 9.5-5.09-5.09 1.41-1.41 3.67 3.68Z"
};
var rotate_90_degrees_ccw = {
  name: "rotate_90_degrees_ccw",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M13.57 4.62c2.3 0 4.61.88 6.36 2.64a8.98 8.98 0 0 1 0 12.72 8.95 8.95 0 0 1-6.36 2.64c-1.49 0-2.98-.38-4.33-1.12l1.49-1.49a6.973 6.973 0 0 0 7.79-1.44 7.007 7.007 0 0 0 0-9.9 6.973 6.973 0 0 0-4.95-2.05v3.24L9.33 5.62l4.24-4.24v3.24ZM7.91 7.03l-6.48 6.49L7.92 20l6.49-6.48-6.5-6.49Zm-3.65 6.49 3.66-3.66 3.65 3.66-3.66 3.66-3.65-3.66Z"
};
var rotate_left = {
  name: "rotate_left",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12.965 2.535v3.07c3.95.49 7 3.85 7 7.93s-3.05 7.44-7 7.93v-2.02c2.84-.48 5-2.94 5-5.91s-2.16-5.43-5-5.91v3.91l-4.55-4.45 4.55-4.55Zm-7.3 6.11 1.41 1.42c-.53.75-.88 1.6-1.02 2.47h-2.02c.17-1.39.73-2.73 1.63-3.89Zm-1.63 5.89h2.02c.14.88.49 1.72 1.01 2.47l-1.41 1.42c-.9-1.16-1.45-2.5-1.62-3.89Zm3.03 5.32c1.16.9 2.51 1.44 3.9 1.61v-2.03c-.87-.15-1.71-.49-2.46-1.03l-1.44 1.45Z"
};
var flip = {
  name: "flip",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M13 23h-2V1h2v22ZM3 19V5c0-1.1.9-2 2-2h4v2H5v14h4v2H5c-1.1 0-2-.9-2-2ZM19 9h2V7h-2v2Zm-4 12h2v-2h-2v2Zm4-18v2h2c0-1.1-.9-2-2-2Zm0 14h2v-2h-2v2ZM17 5h-2V3h2v2Zm2 8h2v-2h-2v2Zm2 6c0 1.1-.9 2-2 2v-2h2Z"
};
var edit_text = {
  name: "edit_text",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17.659 3c-.25 0-.51.1-.7.29l-1.83 1.83 3.75 3.75 1.83-1.83a.996.996 0 0 0 0-1.41l-2.34-2.34c-.2-.2-.45-.29-.71-.29Zm-3.6 6.02.92.92L5.919 19h-.92v-.92l9.06-9.06Zm-11.06 8.23 11.06-11.06 3.75 3.75L6.749 21h-3.75v-3.75Z"
};
var crop_rotate = {
  name: "crop_rotate",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11.416.038c.21-.02.413-.038.634-.038C18.34 0 23.49 4.84 24 11h-1.5c-.36-3.76-2.7-6.93-5.97-8.48L15.2 3.85 11.39.04l.026-.002ZM1.5 13c.36 3.76 2.7 6.93 5.97 8.49l1.33-1.34 3.81 3.82c-.073.003-.146.008-.218.012a7.058 7.058 0 0 1-.442.018C5.66 24 .51 19.16 0 13h1.5ZM16 14h2V8a2 2 0 0 0-2-2h-6v2h6v6ZM8 4v12h12v2h-2v2h-2v-2H8a2 2 0 0 1-2-2V8H4V6h2V4h2Z"
};
var crop = {
  name: "crop",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 1v16h16v2h-4v4h-2v-4H7c-1.1 0-2-.9-2-2V7H1V5h4V1h2Zm12 14h-2V7H9V5h8c1.1 0 2 .9 2 2v8Z"
};
var rotate_right = {
  name: "rotate_right",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m11.035 2.535 4.55 4.55-4.55 4.45v-3.91c-2.84.48-5 2.94-5 5.91s2.16 5.43 5 5.91v2.02c-3.95-.49-7-3.85-7-7.93s3.06-7.44 7-7.93v-3.07Zm7.31 6.11c.9 1.16 1.45 2.5 1.62 3.89h-2.02c-.14-.87-.48-1.72-1.02-2.47l1.42-1.42Zm-5.31 10.79v2.02c1.39-.17 2.74-.71 3.9-1.61l-1.44-1.44c-.75.54-1.59.89-2.46 1.03Zm5.31-1.01-1.42-1.41c.54-.76.88-1.61 1.02-2.48h2.02a7.906 7.906 0 0 1-1.62 3.89Z"
};
var video_chat = {
  name: "video_chat",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20 2H4c-1.1 0-1.99.9-1.99 2L2 22l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2Zm0 14H5.17L4 17.17V4h16v12Zm-3-3-3-2.4V13H7V7h7v2.4L17 7v6Z"
};
var comment_important = {
  name: "comment_important",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 2h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H6l-4 4V4c0-1.1.9-2 2-2Zm1.17 14H20V4H4v13.17L5.17 16ZM11 12h2v2h-2v-2Zm2-6h-2v4h2V6Z"
};
var comment_more = {
  name: "comment_more",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2Zm0 14H5.17L4 17.17V4h16v12ZM9 9H7v2h2V9Zm6 0h2v2h-2V9Zm-2 0h-2v2h2V9Z"
};
var email_draft = {
  name: "email_draft",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M21.99 9.5c0-.72-.37-1.35-.94-1.7L12 2.5 2.95 7.8c-.57.35-.95.98-.95 1.7v10c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2l-.01-10Zm-2 0v.01L12 14.5l-8-5 8-4.68 7.99 4.68ZM4 11.84v7.66h16l-.01-7.63L12 16.86l-8-5.02Z"
};
var share_screen_off = {
  name: "share_screen_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M1.11 2.84 2.66 4.4c-.41.37-.66.89-.66 1.48v9.98c0 1.1.9 2 2.01 2H0v2h18.13l2.71 2.71 1.41-1.41L2.52 1.43 1.11 2.84Zm20.68 15.02 2 2H24v-2h-2.21ZM4.13 5.88H4v10h10.13l-3.46-3.48c-1.54.38-2.71 1.17-3.67 2.46.31-1.48.94-2.93 2.08-4.05L4.13 5.88Zm15.87 0v10.19l1.3 1.3c.42-.37.7-.89.7-1.49v-10a2 2 0 0 0-2-2H7.8l2 2H20Zm-4.28 5.91-2.79-2.78.07-.02V6.86l4 3.73-1.28 1.2Z"
};
var share_screen = {
  name: "share_screen",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20 18c1.1 0 1.99-.9 1.99-2L22 6a2 2 0 0 0-2-2H4c-1.11 0-2 .89-2 2v10a2 2 0 0 0 2 2H0v2h24v-2h-4ZM4 16V6h16v10.01L4 16Zm3-1c.56-2.67 2.11-5.33 6-5.87V7l4 3.73-4 3.74v-2.19c-2.78 0-4.61.85-6 2.72Z"
};
var unsubscribe = {
  name: "unsubscribe",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20.488 5.498v9.04c.78.79 1.19 1.94.94 3.18-.27 1.34-1.36 2.44-2.7 2.71-2.08.42-3.9-1.01-4.18-2.93H4.498c-1.1 0-2-.9-2-2v-10c0-1.1.9-2 2-2h13.99c1.1 0 2 .9 2 2Zm-8.99 3.5 6.99-3.5H4.498l7 3.5Zm3.35 6.5H4.498v-8l7 3.5 7-3.5v6.05l-.108-.014a2.886 2.886 0 0 0-.392-.036c-1.39 0-2.59.82-3.15 2Zm1.15 1v1h4v-1h-4Z"
};
var email = {
  name: "email",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M22 6c0-1.1-.9-2-2-2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6Zm-2 0-8 5-8-5h16Zm-8 7L4 8v10h16V8l-8 5Z"
};
var dialpad = {
  name: "dialpad",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 3c0-1.1.9-2 2-2s2 .9 2 2-.9 2-2 2-2-.9-2-2Zm6 18c0-1.1.9-2 2-2s2 .9 2 2-.9 2-2 2-2-.9-2-2ZM6 7c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2Zm-2 8c0-1.1.9-2 2-2s2 .9 2 2-.9 2-2 2-2-.9-2-2ZM18 5c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2Zm-8 10c0-1.1.9-2 2-2s2 .9 2 2-.9 2-2 2-2-.9-2-2Zm8-2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2Zm-2-4c0-1.1.9-2 2-2s2 .9 2 2-.9 2-2 2-2-.9-2-2Zm-4-2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2Zm-2-4c0-1.1.9-2 2-2s2 .9 2 2-.9 2-2 2-2-.9-2-2Z"
};
var contacts = {
  name: "contacts",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 0h16v2H4V0Zm0 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2Zm0 14h16V6H4v12Zm0 4v2h16v-2H4Zm8-10a2.5 2.5 0 0 0 0-5 2.5 2.5 0 0 0 0 5Zm1-2.5c0-.55-.45-1-1-1s-1 .45-1 1 .45 1 1 1 1-.45 1-1Zm4 6.49C17 13.9 13.69 13 12 13s-5 .9-5 2.99V17h10v-1.01Zm-5-1.49c-1.16 0-2.58.48-3.19 1h6.39c-.61-.52-2.03-1-3.2-1Z"
};
var contact_phone = {
  name: "contact_phone",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M22 3H2C.9 3 0 3.9 0 5v14c0 1.1.9 2 2 2h20c1.1 0 1.99-.9 1.99-2L24 5c0-1.1-.9-2-2-2ZM2 19V5h20v14H2Zm19-3-1.99 1.99A7.512 7.512 0 0 1 16.28 14c-.18-.64-.28-1.31-.28-2s.1-1.36.28-2a7.474 7.474 0 0 1 2.73-3.99L21 8l-1.51 2h-1.64c-.22.63-.35 1.3-.35 2s.13 1.37.35 2h1.64L21 16ZM9 12c1.65 0 3-1.35 3-3s-1.35-3-3-3-3 1.35-3 3 1.35 3 3 3Zm1-3c0-.55-.45-1-1-1s-1 .45-1 1 .45 1 1 1 1-.45 1-1Zm5 7.59c0-2.5-3.97-3.58-6-3.58s-6 1.08-6 3.58V18h12v-1.41ZM9 15c-1.3 0-2.78.5-3.52 1h7.04c-.75-.51-2.22-1-3.52-1Z"
};
var contact_email = {
  name: "contact_email",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 3h20c1.1 0 2 .9 2 2l-.01 14c0 1.1-.89 2-1.99 2H2c-1.1 0-2-.9-2-2V5c0-1.1.9-2 2-2Zm0 16h20V5H2v14ZM21 6h-7v5h7V6Zm-3.5 3.75L20 8V7l-2.5 1.75L15 7v1l2.5 1.75ZM9 12c1.65 0 3-1.35 3-3s-1.35-3-3-3-3 1.35-3 3 1.35 3 3 3Zm1-3c0-.55-.45-1-1-1s-1 .45-1 1 .45 1 1 1 1-.45 1-1Zm5 7.59c0-2.5-3.97-3.58-6-3.58s-6 1.08-6 3.58V18h12v-1.41ZM9 15c-1.3 0-2.78.5-3.52 1h7.04c-.75-.51-2.22-1-3.52-1Z"
};
var email_alpha = {
  name: "email_alpha",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10v1.43C22 15.4 20.47 17 18.5 17c-1.19 0-2.31-.58-2.96-1.47-.9.91-2.16 1.47-3.54 1.47-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5v1.43c0 .79.71 1.57 1.5 1.57s1.5-.78 1.5-1.57V12c0-4.34-3.66-8-8-8s-8 3.66-8 8 3.66 8 8 8h5v2h-5C6.48 22 2 17.52 2 12Zm7 0c0 1.66 1.34 3 3 3s3-1.34 3-3-1.34-3-3-3-3 1.34-3 3Z"
};
var call_add = {
  name: "call_add",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16.43 14.93c1.12.37 2.32.57 3.57.57.55 0 1 .45 1 1V20c0 .55-.45 1-1 1-9.39 0-17-7.61-17-17 0-.55.45-1 1-1h3.5c.55 0 1 .45 1 1 0 1.25.2 2.45.57 3.57.11.35.03.74-.25 1l-2.2 2.21c1.44 2.84 3.76 5.15 6.59 6.59l2.2-2.2c.2-.19.45-.29.71-.29.1 0 .21.02.31.05ZM6.98 7.58c-.23-.83-.38-1.7-.45-2.58h-1.5c.09 1.32.35 2.58.75 3.79l1.2-1.21ZM19 18.97c-1.32-.09-2.6-.35-3.8-.76l1.2-1.2c.85.24 1.72.39 2.6.45v1.51ZM18 3v3h3v2h-3v3h-2V8h-3V6h3V3h2Z"
};
var call_end = {
  name: "call_end",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M.29 12.24C3.34 9.35 7.46 7.57 12 7.57c4.54 0 8.66 1.78 11.71 4.67.18.18.29.43.29.71 0 .28-.11.53-.29.71l-2.48 2.48c-.18.18-.43.29-.71.29-.27 0-.52-.11-.7-.28a11.27 11.27 0 0 0-2.67-1.85.996.996 0 0 1-.56-.9v-3.1c-1.44-.48-2.99-.73-4.59-.73-1.6 0-3.15.25-4.6.72v3.1c0 .39-.23.74-.56.9-.98.49-1.87 1.12-2.66 1.85-.18.18-.43.28-.7.28-.28 0-.53-.11-.71-.29L.29 13.65a.956.956 0 0 1-.29-.7c0-.28.11-.53.29-.71Zm5.11-1.15v1.7c-.65.37-1.28.79-1.87 1.27l-1.07-1.07c.91-.75 1.9-1.38 2.94-1.9Zm13.19 0 .01.005V12.8c.67.38 1.3.8 1.88 1.27L21.55 13a14.798 14.798 0 0 0-2.95-1.905v-.005h-.01Z"
};
var call = {
  name: "call",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 3h3.5c.55 0 1 .45 1 1 0 1.25.2 2.45.57 3.57.11.35.03.74-.25 1.02l-2.2 2.2c1.44 2.83 3.76 5.14 6.59 6.59l2.2-2.2c.2-.19.45-.29.71-.29.1 0 .21.01.31.05 1.12.37 2.33.57 3.57.57.55 0 1 .45 1 1V20c0 .55-.45 1-1 1-9.39 0-17-7.61-17-17 0-.55.45-1 1-1Zm2.54 2c.06.89.21 1.76.45 2.59l-1.2 1.2c-.41-1.2-.67-2.47-.76-3.79h1.51Zm9.86 12.02c.85.24 1.72.39 2.6.45v1.49c-1.32-.09-2.59-.35-3.8-.75l1.2-1.19Z"
};
var comment_discussion = {
  name: "comment_discussion",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16 2H3c-.55 0-1 .45-1 1v14l4-4h10c.55 0 1-.45 1-1V3c0-.55-.45-1-1-1Zm-1 2v7H5.17L4 12.17V4h11Zm4 2h2c.55 0 1 .45 1 1v15l-4-4H7c-.55 0-1-.45-1-1v-2h13V6Z"
};
var comment_chat = {
  name: "comment_chat",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 2c-1.1 0-1.99.9-1.99 2L2 22l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2H4Zm0 2h16v12H5.17L4 17.17V4Zm10 8H6v2h8v-2ZM6 9h12v2H6V9Zm12-3H6v2h12V6Z"
};
var comment_solid = {
  name: "comment_solid",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2Z"
};
var comment = {
  name: "comment",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 2h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H6l-4 4V4c0-1.1.9-2 2-2Zm2 14h14V4H4v14l2-2Z"
};
var mood_very_happy = {
  name: "mood_very_happy",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2ZM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8Zm5-10.5c0 .83-.67 1.5-1.5 1.5S14 10.33 14 9.5 14.67 8 15.5 8s1.5.67 1.5 1.5ZM8.5 11c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11Zm8.61 3c-.8 2.04-2.78 3.5-5.11 3.5-2.33 0-4.31-1.46-5.11-3.5h10.22Z"
};
var mood_very_sad = {
  name: "mood_very_sad",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2ZM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8Zm5-10.5c0 .83-.67 1.5-1.5 1.5S14 10.33 14 9.5 14.67 8 15.5 8s1.5.67 1.5 1.5ZM8.5 11c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11Zm-1.61 6c.8-2.04 2.78-3.5 5.11-3.5 2.33 0 4.31 1.46 5.11 3.5H6.89Z"
};
var mood_sad = {
  name: "mood_sad",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2ZM8.5 8a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm7 0a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm-8.62 9.5a5.495 5.495 0 0 1 10.24 0h-1.67c-.7-1.19-1.97-2-3.45-2-1.48 0-2.76.81-3.45 2H6.88ZM4 12c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8-8 3.58-8 8Z"
};
var mood_neutral = {
  name: "mood_neutral",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2ZM7 9.5a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0ZM15.5 8a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3ZM9 15.5V14h6v1.5H9ZM4 12c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8-8 3.58-8 8Z"
};
var mood_happy = {
  name: "mood_happy",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2ZM8.5 8a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm7 0a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm-6.95 6c.7 1.19 1.97 2 3.45 2 1.48 0 2.75-.81 3.45-2h1.67a5.495 5.495 0 0 1-10.24 0h1.67ZM4 12c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8-8 3.58-8 8Z"
};
var mood_extremely_sad = {
  name: "mood_extremely_sad",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 12C2 6.47 6.47 2 11.99 2 17.52 2 22 6.47 22 12s-4.49 10-10.01 10S2 17.53 2 12Zm5.82 0 1.06-1.06L9.94 12 11 10.94 9.94 9.88 11 8.82 9.94 7.76 8.88 8.82 7.82 7.76 6.76 8.82l1.06 1.06-1.06 1.06L7.82 12ZM12 13.5c-2.33 0-4.31 1.46-5.11 3.5h10.22c-.8-2.04-2.78-3.5-5.11-3.5Zm0 6.5c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8Zm3.12-11.18 1.06-1.06 1.06 1.06-1.06 1.06 1.06 1.06L16.18 12l-1.06-1.06L14.06 12 13 10.94l1.06-1.06L13 8.82l1.06-1.06 1.06 1.06Z"
};
var mood_extremely_happy = {
  name: "mood_extremely_happy",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11.99 2C6.47 2 2 6.47 2 12s4.47 10 9.99 10C17.51 22 22 17.53 22 12S17.52 2 11.99 2ZM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8Zm2.06-9L13 9.94l2.12-2.12 2.12 2.12L16.18 11l-1.06-1.06L14.06 11ZM8.88 9.94 9.94 11 11 9.94 8.88 7.82 6.76 9.94 7.82 11l1.06-1.06ZM17.11 14c-.8 2.04-2.78 3.5-5.11 3.5-2.33 0-4.31-1.46-5.11-3.5h10.22Z"
};
var comment_add = {
  name: "comment_add",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M22 4c0-1.1-.9-2-2-2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h14l4 4V4Zm-2 13.17L18.83 16H4V4h16v13.17ZM11 5h2v4h4v2h-4v4h-2v-4H7V9h4V5Z"
};
var comment_chat_off = {
  name: "comment_chat_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2.34.93.93 2.34l2.01 2.01-.01 16.99 4-4h9l5.73 5.73 1.41-1.41L2.34.93Zm18.59 14.41v-12H7.59l-2-2h15.34c1.1 0 2 .9 2 2v12c0 .9-.61 1.66-1.43 1.91l-1.91-1.91h1.34Zm-12-4h-2v2h2v-2Zm10-3h-6.34l2 2h4.34v-2Zm-8-3h8v2h-7.34l-.66-.66V5.34Zm-6 11.17 1.17-1.17h7.83l-5-5h-2v-2l-2-2v10.17Z"
};
var comment_notes = {
  name: "comment_notes",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 2h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H6l-4 4 .01-18c0-1.1.89-2 1.99-2Zm1.17 14H20V4H4v13.17l.58-.58.59-.59ZM6 12h2v2H6v-2Zm2-3H6v2h2V9ZM6 6h2v2H6V6Zm9 6h-5v2h5v-2Zm-5-3h8v2h-8V9Zm8-3h-8v2h8V6Z"
};
var mail_unread = {
  name: "mail_unread",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20 7H10v2h10v12H4V9h2v4h2V5h6V1H6v6H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V9c0-1.1-.9-2-2-2Z"
};
var thumbs_down = {
  name: "thumbs_down",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15 2H6c-.83 0-1.54.5-1.84 1.22l-3.02 7.05c-.09.23-.14.47-.14.73v2c0 1.1.9 2 2 2h6.31l-.95 4.57-.03.32c0 .41.17.79.44 1.06L9.83 22l6.59-6.59c.36-.36.58-.86.58-1.41V4c0-1.1-.9-2-2-2Zm0 12-4.34 4.34L12 13H3v-2l3-7h9v10Zm8-12h-4v12h4V2Z"
};
var thumbs_up_down = {
  name: "thumbs_up_down",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11 5c.55 0 1 .45 1 1v1.25c0 .19-.04.38-.11.55l-2.26 5.29c-.23.53-.76.91-1.38.91H1.5C.67 14 0 13.33 0 12.5V6c0-.41.17-.79.44-1.06L5.38 0l.79.79c.2.21.33.49.33.8l-.02.23L5.82 5H11Zm-3.08 7L10 7.13V7H3.36l.57-2.72L2 6.21V12h5.92Zm14.58-2h-6.75c-.62 0-1.15.38-1.38.91l-2.26 5.29c-.07.17-.11.36-.11.55V18c0 .55.45 1 1 1h5.18l-.66 3.18-.02.24c0 .31.13.59.33.8l.79.78 4.94-4.94c.27-.27.44-.65.44-1.06v-6.5c0-.83-.67-1.5-1.5-1.5Zm-2.43 9.72L22 17.79V12h-5.92L14 16.87V17h6.64l-.57 2.72Z"
};
var support = {
  name: "support",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11.5 19.695v3.6l1.43-.69c5.13-2.47 8.57-7.45 8.57-12.4 0-5.24-4.26-9.5-9.5-9.5s-9.5 4.26-9.5 9.5c0 5.07 3.99 9.23 9 9.49Zm-7-9.49c0-4.14 3.36-7.5 7.5-7.5 4.14 0 7.5 3.36 7.5 7.5 0 3.72-2.36 7.5-6 9.8v-2.3H12c-4.14 0-7.5-3.36-7.5-7.5Zm8.5 4v2h-2v-2h2Zm-.23-4.678c-.828.635-1.77 1.357-1.77 3.178h2c0-1.095.711-1.717 1.44-2.354.77-.673 1.56-1.363 1.56-2.646 0-2.21-1.79-4-4-4s-4 1.79-4 4h2c0-1.1.9-2 2-2s2 .9 2 2c0 .88-.58 1.324-1.23 1.822Z"
};
var help = {
  name: "help",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2Zm-1 17v-2h2v2h-2Zm3.17-6.83.9-.92c.57-.57.93-1.37.93-2.25 0-2.21-1.79-4-4-4S8 6.79 8 9h2c0-1.1.9-2 2-2s2 .9 2 2c0 .55-.22 1.05-.59 1.41l-1.24 1.26C11.45 12.4 11 13.4 11 14.5v.5h2c0-1.5.45-2.1 1.17-2.83Z"
};
var help_outline = {
  name: "help_outline",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10-4.48 10-10 10S2 17.52 2 12Zm11 4v2h-2v-2h2Zm-1 4c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8ZM8 10c0-2.21 1.79-4 4-4s4 1.79 4 4c0 1.283-.79 1.973-1.56 2.646C13.712 13.283 13 13.905 13 15h-2c0-1.821.942-2.543 1.77-3.178.65-.498 1.23-.943 1.23-1.822 0-1.1-.9-2-2-2s-2 .9-2 2H8Z"
};
var info_circle = {
  name: "info_circle",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12.01 22C17.53 22 22 17.52 22 12S17.53 2 12.01 2C6.48 2 2 6.48 2 12s4.48 10 10.01 10ZM13 9V7h-2v2h2Zm0 8v-6h-2v6h2Zm7-5c0-4.42-3.58-8-8-8s-8 3.58-8 8 3.58 8 8 8 8-3.58 8-8Z"
};
var thumbs_up = {
  name: "thumbs_up",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M9 22h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-2c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L14.17 2 7.58 8.59C7.22 8.95 7 9.45 7 10v10c0 1.1.9 2 2 2Zm0-12 4.34-4.34L12 11h9v2l-3 7H9V10Zm-4 0H1v12h4V10Z",
  sizes: {
    small: {
      name: "thumbs_up_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M5.625 17h7.313c.674 0 1.25-.4 1.494-.976l2.454-5.64c.073-.184.114-.376.114-.584V8.2c0-.88-.606-1.2-1.5-1.2H10l1.02-4.056.024-.256c0-.328-.138-.632-.357-.848L9.826 1 4.47 6.272A1.578 1.578 0 0 0 4 7.4v8c0 .88.731 1.6 1.625 1.6ZM6 7.5l3.5-4L8 9h7v1l-2 5H6V7.5ZM3 7H1v10h2V7Z"
    }
  }
};
var donut_large = {
  name: "donut_large",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5.025 12c0 3.52 2.61 6.43 6 6.92v3.03c-5.05-.5-9-4.76-9-9.95 0-5.19 3.95-9.45 9-9.95v3.03c-3.39.49-6 3.4-6 6.92Zm8-6.92a7 7 0 0 1 5.92 5.92h3.03c-.47-4.72-4.23-8.48-8.95-8.95v3.03Zm5.92 7.92a7 7 0 0 1-5.92 5.92v3.03c4.72-.47 8.48-4.23 8.95-8.95h-3.03Z"
};
var donut_outlined = {
  name: "donut_outlined",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2.025 12c0 5.19 3.95 9.45 9 9.95v-7.13c-1.16-.42-2-1.52-2-2.82 0-1.3.84-2.4 2-2.82V2.05c-5.05.5-9 4.76-9 9.95Zm19.95-1h-7.13c-.31-.85-.97-1.51-1.82-1.82V2.05c4.72.47 8.48 4.23 8.95 8.95Zm-2.53-2c-.82-2-2.42-3.6-4.42-4.42v3.43c.37.28.71.62.99.99h3.43Zm-10.42-.98V4.58c-2.96 1.18-5 4.07-5 7.42 0 3.35 2.04 6.24 5 7.43v-3.44c-1.23-.93-2-2.4-2-3.99 0-1.59.77-3.06 2-3.98Zm4 6.8v7.13c4.72-.47 8.48-4.23 8.95-8.95h-7.13c-.31.85-.97 1.51-1.82 1.82Zm2.99.18c-.28.38-.62.71-.99.99v3.43c2-.82 3.6-2.42 4.42-4.42h-3.43Z"
};
var table_chart = {
  name: "table_chart",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19.5 3h-15c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h15c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2Zm0 2v3h-15V5h15Zm-10 14h5v-9h-5v9Zm-5-9h3v9h-3v-9Zm12 0v9h3v-9h-3Z"
};
var multiline_chart = {
  name: "multiline_chart",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m22 7.175-1.41-1.41-2.85 3.21c-2.06-2.32-4.91-3.72-8.13-3.72-2.89 0-5.54 1.16-7.61 3l1.42 1.42c1.7-1.49 3.85-2.42 6.19-2.42 2.74 0 5.09 1.26 6.77 3.24l-2.88 3.24-4-4-7.5 7.51 1.5 1.5 6-6.01 4 4 4.05-4.55c.75 1.35 1.25 2.9 1.44 4.55H21c-.22-2.3-.95-4.39-2.04-6.14L22 7.175Z"
};
var pie_chart = {
  name: "pie_chart",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10-4.48 10-10 10S2 17.52 2 12Zm11-1h6.93A8.002 8.002 0 0 0 13 4.07V11Zm-9 1c0-4.07 3.06-7.44 7-7.93v15.86c-3.94-.49-7-3.86-7-7.93Zm9 1v6.93A8.002 8.002 0 0 0 19.93 13H13Z"
};
var bubble_chart = {
  name: "bubble_chart",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16 3a5.51 5.51 0 0 0-5.5 5.5c0 3.03 2.47 5.5 5.5 5.5s5.5-2.47 5.5-5.5S19.03 3 16 3ZM2.5 14c0-2.21 1.79-4 4-4s4 1.79 4 4-1.79 4-4 4-4-1.79-4-4Zm2 0c0 1.1.9 2 2 2s2-.9 2-2-.9-2-2-2-2 .9-2 2Zm10.01 1c-1.65 0-3 1.35-3 3s1.35 3 3 3 3-1.35 3-3-1.35-3-3-3Zm-1 3c0 .55.45 1 1 1s1-.45 1-1-.45-1-1-1-1 .45-1 1ZM12.5 8.5c0 1.93 1.57 3.5 3.5 3.5s3.5-1.57 3.5-3.5S17.93 5 16 5s-3.5 1.57-3.5 3.5Z"
};
var scatter_plot = {
  name: "scatter_plot",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11.2 10.2c-2.21 0-4-1.79-4-4s1.79-4 4-4 4 1.79 4 4-1.79 4-4 4Zm-8 4c0 2.21 1.79 4 4 4s4-1.79 4-4-1.79-4-4-4-4 1.79-4 4Zm2 0c0-1.1.9-2 2-2s2 .9 2 2-.9 2-2 2-2-.9-2-2Zm4-8c0-1.1.9-2 2-2s2 .9 2 2-.9 2-2 2-2-.9-2-2Zm7.6 15.6c-2.21 0-4-1.79-4-4s1.79-4 4-4 4 1.79 4 4-1.79 4-4 4Zm-2-4c0-1.1.9-2 2-2s2 .9 2 2-.9 2-2 2-2-.9-2-2Z"
};
var bar_chart = {
  name: "bar_chart",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M10.6 5h2.8v14h-2.8V5ZM5 9.2h3V19H5V9.2ZM19 13h-2.8v6H19v-6Z"
};
var assignment = {
  name: "assignment",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14.82 4H19c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H5c-.14 0-.27-.01-.4-.03a2.008 2.008 0 0 1-1.44-1.19c-.1-.24-.16-.51-.16-.78V6c0-.28.06-.54.16-.77A2.008 2.008 0 0 1 4.6 4.04c.13-.03.26-.04.4-.04h4.18C9.6 2.84 10.7 2 12 2c1.3 0 2.4.84 2.82 2ZM7 10V8h10v2H7Zm10 4v-2H7v2h10Zm-3 2H7v2h7v-2ZM12 3.75c.41 0 .75.34.75.75s-.34.75-.75.75-.75-.34-.75-.75.34-.75.75-.75ZM5 20h14V6H5v14Z"
};
var assignment_user = {
  name: "assignment_user",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 4h-4.18C14.4 2.84 13.3 2 12 2c-1.3 0-2.4.84-2.82 2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2Zm-7-.25c.22 0 .41.1.55.25.12.13.2.31.2.5 0 .41-.34.75-.75.75s-.75-.34-.75-.75c0-.19.08-.37.2-.5.14-.15.33-.25.55-.25ZM5 6v14h14V6H5Zm7 1c-1.65 0-3 1.35-3 3s1.35 3 3 3 3-1.35 3-3-1.35-3-3-3Zm-1 3c0 .55.45 1 1 1s1-.45 1-1-.45-1-1-1-1 .45-1 1Zm-5 7.47V19h12v-1.53c0-2.5-3.97-3.58-6-3.58s-6 1.07-6 3.58Zm6-1.59c-1.31 0-3 .56-3.69 1.12h7.38c-.68-.56-2.38-1.12-3.69-1.12Z"
};
var assignment_important = {
  name: "assignment_important",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14.82 4H19c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H5c-.14 0-.27-.01-.4-.03a2.008 2.008 0 0 1-1.44-1.19c-.1-.24-.16-.51-.16-.78V6c0-.28.06-.54.16-.77A2.008 2.008 0 0 1 4.6 4.04c.13-.03.26-.04.4-.04h4.18C9.6 2.84 10.7 2 12 2c1.3 0 2.4.84 2.82 2ZM11 14V8h2v6h-2Zm0 4v-2h2v2h-2Zm1-14.25c.41 0 .75.34.75.75s-.34.75-.75.75-.75-.34-.75-.75.34-.75.75-.75ZM5 20h14V6H5v14Z"
};
var assignment_return = {
  name: "assignment_return",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14.82 4H19c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H5c-.14 0-.27-.01-.4-.03a2.008 2.008 0 0 1-1.44-1.19c-.1-.24-.16-.51-.16-.78V6c0-.28.06-.54.16-.77A2.008 2.008 0 0 1 4.6 4.04c.13-.03.26-.04.4-.04h4.18C9.6 2.84 10.7 2 12 2c1.3 0 2.4.84 2.82 2ZM16 15h-4v3l-5-5 5-5v3h4v4ZM12 3.75c.41 0 .75.34.75.75s-.34.75-.75.75-.75-.34-.75-.75.34-.75.75-.75ZM5 20h14V6H5v14Z"
};
var assignment_returned = {
  name: "assignment_returned",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14.82 4H19c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H5c-.14 0-.27-.01-.4-.03a2.008 2.008 0 0 1-1.44-1.19c-.1-.24-.16-.51-.16-.78V6c0-.28.06-.54.16-.77A2.008 2.008 0 0 1 4.6 4.04c.13-.03.26-.04.4-.04h4.18C9.6 2.84 10.7 2 12 2c1.3 0 2.4.84 2.82 2ZM14 13h3l-5 5-5-5h3V9h4v4Zm-2-9.25c.41 0 .75.34.75.75s-.34.75-.75.75-.75-.34-.75-.75.34-.75.75-.75ZM5 20h14V6H5v14Z"
};
var assignment_turned_in = {
  name: "assignment_turned_in",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14.82 4H19c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H5c-.14 0-.27-.01-.4-.03a2.008 2.008 0 0 1-1.44-1.19c-.1-.24-.16-.51-.16-.78V6c0-.28.06-.54.16-.77A2.008 2.008 0 0 1 4.6 4.04c.13-.03.26-.04.4-.04h4.18C9.6 2.84 10.7 2 12 2c1.3 0 2.4.84 2.82 2Zm1.77 4.58L18 10l-8 8-4-4 1.41-1.41L10 15.17l6.59-6.59ZM12 3.75c.41 0 .75.34.75.75s-.34.75-.75.75-.75-.34-.75-.75.34-.75.75-.75ZM5 20h14V6H5v14Z"
};
var timeline = {
  name: "timeline",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M23 8c0 1.1-.9 2-2 2a1.7 1.7 0 0 1-.51-.07l-3.56 3.55c.05.16.07.34.07.52 0 1.1-.9 2-2 2s-2-.9-2-2c0-.18.02-.36.07-.52l-2.55-2.55c-.16.05-.34.07-.52.07s-.36-.02-.52-.07l-4.55 4.56c.05.16.07.33.07.51 0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2c.18 0 .35.02.51.07l4.56-4.55C8.02 9.36 8 9.18 8 9c0-1.1.9-2 2-2s2 .9 2 2c0 .18-.02.36-.07.52l2.55 2.55c.16-.05.34-.07.52-.07s.36.02.52.07l3.55-3.56A1.7 1.7 0 0 1 19 8c0-1.1.9-2 2-2s2 .9 2 2Z"
};
var trending_down = {
  name: "trending_down",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m16 18 2.29-2.29-4.88-4.88-4 4L2 7.41 3.41 6l6 6 4-4 6.3 6.29L22 12v6h-6Z"
};
var trending_flat = {
  name: "trending_flat",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m21.5 12-4-4v3h-15v2h15v3l4-4Z"
};
var trending_up = {
  name: "trending_up",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m16 6 2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6h-6Z"
};
var eject = {
  name: "eject",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 5 5.33 15h13.34L12 5Zm7 12v2H5v-2h14Zm-4.07-4L12 8.6 9.07 13h5.86Z"
};
var music_note = {
  name: "music_note",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 13.55V3h6v4h-4v10c0 2.21-1.79 4-4 4s-4-1.79-4-4 1.79-4 4-4c.73 0 1.41.21 2 .55ZM8 17c0 1.1.9 2 2 2s2-.9 2-2-.9-2-2-2-2 .9-2 2Z"
};
var music_note_off = {
  name: "music_note_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11.93 10.51 4.34 2.93 2.93 4.34l9 9v.28c-.94-.54-2.1-.75-3.33-.32-1.34.48-2.37 1.67-2.61 3.07a4.007 4.007 0 0 0 4.59 4.65c1.96-.31 3.35-2.11 3.35-4.1v-1.58l5.73 5.73 1.41-1.41-9.14-9.15Zm2-3.44h4v-4h-6v4.61l2 2V7.07Zm-6 10c0 1.1.9 2 2 2s2-.9 2-2-.9-2-2-2-2 .9-2 2Z"
};
var mic = {
  name: "mic",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15 11.5c0 1.66-1.34 3-3 3s-3-1.34-3-3v-6c0-1.66 1.34-3 3-3s3 1.34 3 3v6Zm-3 5c2.76 0 5-2.24 5-5h2c0 3.53-2.61 6.43-6 6.92v3.08h-2v-3.08c-3.39-.49-6-3.39-6-6.92h2c0 2.76 2.24 5 5 5Z"
};
var mic_outlined = {
  name: "mic_outlined",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 14.5c1.66 0 3-1.34 3-3v-6c0-1.66-1.34-3-3-3s-3 1.34-3 3v6c0 1.66 1.34 3 3 3Zm-1-9c0-.55.45-1 1-1s1 .45 1 1v6c0 .55-.45 1-1 1s-1-.45-1-1v-6Zm1 11c2.76 0 5-2.24 5-5h2c0 3.53-2.61 6.43-6 6.92v3.08h-2v-3.08c-3.39-.49-6-3.39-6-6.92h2c0 2.76 2.24 5 5 5Z"
};
var mic_off = {
  name: "mic_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M10.72 5.39c0-.66.54-1.2 1.2-1.2.66 0 1.2.54 1.2 1.2l-.01 3.91 1.81 1.79v-5.6c0-1.66-1.34-3-3-3-1.54 0-2.79 1.16-2.96 2.65l1.76 1.76V5.39Zm8.2 6.1h-1.7c0 .58-.1 1.13-.27 1.64l1.27 1.27c.44-.88.7-1.87.7-2.91Zm-16-6.73 1.41-1.41L21.08 20.1l-1.41 1.41-4.2-4.2c-.78.45-1.64.77-2.55.9v3.28h-2v-3.28c-3.28-.49-6-3.31-6-6.72h1.7c0 3 2.54 5.1 5.3 5.1.81 0 1.6-.19 2.31-.52l-1.66-1.66c-.21.05-.42.08-.65.08-1.66 0-3-1.34-3-3v-.73l-6-6Z"
};
var volume_up = {
  name: "volume_up",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14 5.29V3.23c4.01.91 7 4.49 7 8.77 0 4.28-2.99 7.86-7 8.77v-2.06c2.89-.86 5-3.54 5-6.71s-2.11-5.85-5-6.71ZM3 15V9h4l5-5v16l-5-5H3Zm7 .17V8.83L7.83 11H5v2h2.83L10 15.17ZM16.5 12A4.5 4.5 0 0 0 14 7.97v8.05c1.48-.73 2.5-2.25 2.5-4.02Z"
};
var volume_off = {
  name: "volume_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m4.34 2.935-1.41 1.41 4.36 4.36-.29.3H3v6h4l5 5v-6.59l4.18 4.18c-.65.49-1.38.88-2.18 1.11v2.06a8.94 8.94 0 0 0 3.61-1.75l2.05 2.05 1.41-1.41L4.34 2.935ZM10 15.175l-2.17-2.17H5v-2h2.83l.88-.88 1.29 1.29v3.76Zm8.59-.83c.26-.73.41-1.52.41-2.34 0-3.17-2.11-5.85-5-6.71v-2.06c4.01.91 7 4.49 7 8.77 0 1.39-.32 2.7-.88 3.87l-1.53-1.53ZM12 4.005l-1.88 1.88L12 7.765v-3.76Zm2 3.97a4.5 4.5 0 0 1 2.5 4.03c0 .08-.01.16-.02.24L14 9.765v-1.79Z"
};
var volume_mute = {
  name: "volume_mute",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m11.5 9 5-5v16l-5-5h-4V9h4Zm3 6.17V8.83L12.33 11H9.5v2h2.83l2.17 2.17Z"
};
var volume_down = {
  name: "volume_down",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5.25 9v6h4l5 5V4l-5 5h-4Zm11-1.03v8.05c1.48-.73 2.5-2.25 2.5-4.02a4.5 4.5 0 0 0-2.5-4.03Zm-4 7.2V8.83L10.08 11H7.25v2h2.83l2.17 2.17Z"
};
var videocam_off = {
  name: "videocam_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m9.99 8.57-2-2-4.15-4.14-1.41 1.41 2.73 2.73h-.73c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.21 0 .39-.08.55-.18l3.18 3.18 1.41-1.41-8.86-8.86-2.72-2.73Zm-4.56 8v-8h1.73l8 8H5.43Zm10-5.39V8.57h-2.61l-2-2h5.61c.55 0 1 .45 1 1v3.5l4-4v10.11l-6-6Z"
};
var videocam = {
  name: "videocam",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 6h12c.55 0 1 .45 1 1v3.5l4-4v11l-4-4V17c0 .55-.45 1-1 1H4c-.55 0-1-.45-1-1V7c0-.55.45-1 1-1Zm11 10V8H5v8h10Z"
};
var video_call = {
  name: "video_call",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4ZM15 16H5V8h10v8Zm-4-1H9v-2H7v-2h2V9h2v2h2v2h-2v2Z"
};
var missed_video_call = {
  name: "missed_video_call",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4Zm-2-1.83V16H5V8h10v.67ZM11 15l-3.89-3.89v2.55H6V9.22h4.44v1.11H7.89l3.11 3.1 2.99-3.01.78.79L11 15Z"
};
var playlist_play = {
  name: "playlist_play",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M16.5 5h-12v2h12V5Zm0 4h-12v2h12V9Zm-12 4h8v2h-8v-2Zm15 3-5 3v-6l5 3Z"
};
var playlist_added = {
  name: "playlist_added",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M1.5 5h12v2h-12V5Zm0 4h12v2h-12V9Zm0 6h8v-2h-8v2Zm21-3L21 10.5 15.51 16l-3.01-3-1.5 1.5 4.51 4.5 6.99-7Z"
};
var playlist_add = {
  name: "playlist_add",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 5h12v2H2V5Zm0 4h12v2H2V9Zm16 0h-2v4h-4v2h4v4h2v-4h4v-2h-4V9Zm-8 6H2v-2h8v2Z"
};
var snooze = {
  name: "snooze",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m2.055 5.749 1.281 1.536L7.944 3.44 6.663 1.905 2.055 5.75Zm14.001-2.307 1.281-1.536 4.608 3.843-1.282 1.536-4.607-3.843ZM9 11.095h3.63L9 15.295v1.8h6v-2h-3.63l3.63-4.2v-1.8H9v2Zm3-5c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.14-7-7 3.14-7 7-7Zm-9 7A9 9 0 1 1 21 13.097 9 9 0 0 1 3 13.095Z"
};
var shuffle = {
  name: "shuffle",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M10.59 9.17 5.41 4 4 5.41l5.17 5.17 1.42-1.41ZM14.5 4l2.04 2.04L4 18.59 5.41 20 17.96 7.46 20 9.5V4h-5.5Zm-1.08 10.82 1.41-1.41 3.13 3.13L20 14.5V20h-5.5l2.05-2.05-3.13-3.13Z"
};
var repeat_one = {
  name: "repeat_one",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7 7h10v3l4-4-4-4v3H5v6h2V7Zm12 6h-2v4H7v-3l-4 4 4 4v-3h12v-6Zm-6-4v6h-1.5v-4H10v-1l2-1h1Z"
};
var repeat = {
  name: "repeat",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17 7H7v4H5V5h12V2l4 4-4 4V7ZM7 17h10v-4h2v6H7v3l-4-4 4-4v3Z"
};
var movie = {
  name: "movie",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m20 8-2-4h4v14c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2l.01-12c0-1.1.89-2 1.99-2h1l2 4h3L8 4h2l2 4h3l-2-4h2l2 4h3ZM5.76 10 4 6.47V18h16v-8H5.76Z"
};
var res_hd_outlined = {
  name: "res_hd_outlined",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2Zm0 16H5V5h14v14Zm-9.5-6h-2v2H6V9h1.5v2.5h2V9H11v6H9.5v-2Zm8.5 1v-4c0-.55-.45-1-1-1h-4v6h4c.55 0 1-.45 1-1Zm-3.5-.5h2v-3h-2v3Z"
};
var res_hd_filled = {
  name: "res_hd_filled",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 3h14c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Zm4.5 10h-2v2H6V9h1.5v2.5h2V9H11v6H9.5v-2Zm8.5 1v-4c0-.55-.45-1-1-1h-4v6h4c.55 0 1-.45 1-1Zm-3.5-.5h2v-3h-2v3Z"
};
var forward_30 = {
  name: "forward_30",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 20c3.31 0 6-2.69 6-6h2c0 4.42-3.58 8-8 8s-8-3.58-8-8 3.58-8 8-8V2l5 5-5 5V8c-3.31 0-6 2.69-6 6s2.69 6 6 6Zm-1.66-3.66c.08-.03.14-.07.2-.12l.041-.047c.035-.038.067-.072.089-.123.03-.07.04-.15.04-.24 0-.11-.02-.21-.05-.29a.483.483 0 0 0-.14-.2.648.648 0 0 0-.22-.11.885.885 0 0 0-.29-.04h-.45v-.66h.43c.22 0 .37-.05.48-.16.11-.11.16-.25.16-.43 0-.08-.01-.16-.04-.22a.62.62 0 0 0-.11-.17.544.544 0 0 0-.18-.11.625.625 0 0 0-.25-.04c-.08 0-.15 0-.22.03s-.13.06-.18.1a.455.455 0 0 0-.17.35h-.85c0-.18.04-.33.11-.48.07-.15.18-.27.3-.37.12-.1.28-.18.44-.23.16-.05.35-.08.54-.08.22 0 .41.03.59.08s.33.13.46.23.23.22.3.38c.07.16.11.33.11.53 0 .09-.01.18-.04.27-.03.09-.08.17-.13.25-.05.08-.12.15-.2.22-.08.07-.18.12-.28.17.24.09.42.22.54.39.12.17.18.38.18.61 0 .2-.04.38-.12.53a1.1 1.1 0 0 1-.32.39c-.14.1-.29.19-.48.24-.19.05-.39.08-.6.08-.18 0-.36-.02-.53-.07-.17-.05-.32-.13-.46-.23s-.25-.22-.33-.38c-.08-.16-.12-.34-.12-.55h.85c0 .08.02.15.05.22.03.07.07.12.13.17a.69.69 0 0 0 .45.15c.1 0 .19-.01.27-.04Zm4.1-3.56c-.18-.07-.37-.1-.59-.1-.22 0-.41.03-.59.1s-.33.18-.45.33c-.12.15-.23.34-.29.57-.06.23-.1.5-.1.82v.74c0 .32.04.6.11.82.07.22.17.42.3.57.13.15.28.26.46.33s.37.1.59.1c.22 0 .41-.03.59-.1s.33-.18.45-.33c.12-.15.22-.34.29-.57.07-.23.1-.5.1-.82v-.74c0-.32-.04-.6-.11-.82-.07-.22-.17-.42-.3-.57-.13-.15-.28-.26-.46-.33Zm-.03 3.05c.03-.13.04-.29.04-.48h.01v-.97c0-.19-.01-.35-.04-.48s-.07-.23-.12-.31a.436.436 0 0 0-.19-.17.655.655 0 0 0-.5 0c-.08.03-.13.09-.19.17-.06.08-.09.18-.12.31s-.04.29-.04.48v.97c0 .19.01.35.04.48s.07.24.12.32c.05.08.12.14.19.17a.655.655 0 0 0 .5 0c.08-.03.14-.09.19-.17.05-.08.08-.19.11-.32Z"
};
var forward_10 = {
  name: "forward_10",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 20c3.31 0 6-2.69 6-6h2c0 4.42-3.58 8-8 8s-8-3.58-8-8 3.58-8 8-8V2l5 5-5 5V8c-3.31 0-6 2.69-6 6s2.69 6 6 6Zm-1.1-7.27V17h-.85v-3.26l-1.01.31v-.69l1.77-.63h.09Zm3.42.05c-.18-.07-.37-.1-.59-.1-.22 0-.41.03-.59.1s-.33.18-.45.33c-.12.15-.23.34-.29.57-.06.23-.1.5-.1.82v.74c0 .32.04.6.11.82.07.22.17.42.3.57.13.15.28.26.46.33s.37.1.59.1c.22 0 .41-.03.59-.1s.33-.18.45-.33c.12-.15.22-.34.29-.57.07-.23.1-.5.1-.82v-.74c0-.32-.04-.6-.11-.82-.07-.22-.17-.42-.3-.57-.13-.15-.29-.26-.46-.33Zm-.03 3.05c.03-.13.04-.29.04-.48h.01v-.97c0-.19-.01-.35-.04-.48s-.07-.23-.12-.31a.436.436 0 0 0-.19-.17.655.655 0 0 0-.5 0c-.08.03-.13.09-.19.17-.06.08-.09.18-.12.31s-.04.29-.04.48v.97c0 .19.01.35.04.48s.07.24.12.32c.05.08.12.14.19.17a.655.655 0 0 0 .5 0c.08-.03.14-.09.19-.17.05-.08.08-.19.11-.32Z"
};
var forward_5 = {
  name: "forward_5",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 20c3.31 0 6-2.69 6-6h2c0 4.42-3.58 8-8 8s-8-3.58-8-8 3.58-8 8-8V2l5 5-5 5V8c-3.31 0-6 2.69-6 6s2.69 6 6 6Zm.3-3.68c.07-.04.13-.1.18-.17.05-.07.09-.15.11-.24.02-.09.03-.19.03-.31s-.01-.21-.04-.31a.681.681 0 0 0-.13-.24.538.538 0 0 0-.21-.15.853.853 0 0 0-.3-.05c-.07 0-.15.01-.2.02-.05.01-.1.03-.15.05-.05.02-.08.04-.12.07-.04.03-.07.06-.1.09l-.67-.17.25-2.17h2.39v.71h-1.7l-.11.92a.369.369 0 0 0 .048-.021c.017-.009.037-.019.062-.029.025-.01.05-.018.075-.025a.879.879 0 0 0 .075-.025.822.822 0 0 1 .18-.04c.06-.01.13-.02.2-.02.21 0 .39.04.55.1.16.06.3.16.41.28.11.12.19.28.25.45.06.17.09.38.09.6 0 .19-.03.37-.09.54-.06.17-.15.32-.27.45-.12.13-.27.23-.45.31-.18.08-.4.12-.64.12-.18 0-.36-.03-.53-.08-.17-.05-.33-.13-.46-.24-.13-.11-.24-.23-.32-.39-.08-.16-.12-.33-.13-.53h.84c.02.17.08.31.19.41.11.1.25.15.42.15.1 0 .2-.02.27-.06Z"
};
var replay_30 = {
  name: "replay_30",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2v4c4.42 0 8 3.58 8 8s-3.58 8-8 8-8-3.58-8-8h2c0 3.31 2.69 6 6 6s6-2.69 6-6-2.69-6-6-6v4L7 7l5-5Zm-1.99 12.49h-.45v.65h.47c.11 0 .2.02.29.04a.508.508 0 0 1 .36.31c.03.08.05.18.05.29 0 .09-.01.17-.04.24a.561.561 0 0 1-.33.3c-.08.03-.17.04-.27.04-.09 0-.17-.02-.25-.04a.49.49 0 0 1-.2-.11c-.06-.05-.1-.1-.13-.17a.545.545 0 0 1-.05-.22h-.85c0 .21.04.4.12.55.08.15.2.27.33.38.13.11.29.18.46.23.17.05.35.07.53.07.22 0 .41-.03.6-.08s.34-.14.48-.24c.14-.1.24-.24.32-.39.08-.15.12-.33.12-.53 0-.23-.06-.43-.18-.61s-.3-.3-.54-.39a1.1 1.1 0 0 0 .48-.39.853.853 0 0 0 .17-.52c0-.2-.04-.38-.11-.53-.07-.15-.17-.28-.3-.38-.13-.1-.28-.18-.46-.23-.18-.05-.38-.08-.59-.08-.19 0-.37.03-.54.08-.17.05-.31.13-.44.23a1.067 1.067 0 0 0-.41.85h.85a.455.455 0 0 1 .17-.35.4.4 0 0 1 .18-.1.78.78 0 0 1 .22-.03c.09 0 .18.02.25.04s.13.06.18.11a.538.538 0 0 1 .15.39c0 .18-.05.32-.16.43-.11.11-.27.16-.48.16Zm5.29.75c0 .32-.03.6-.1.82-.07.22-.17.42-.29.57-.12.15-.28.26-.45.33-.17.07-.37.1-.59.1-.22 0-.41-.03-.59-.1s-.33-.18-.46-.33c-.13-.15-.23-.34-.3-.57-.07-.23-.11-.5-.11-.82v-.74c0-.32.03-.6.1-.82.07-.22.17-.42.29-.57.12-.15.28-.26.45-.33.17-.07.37-.1.59-.1.22 0 .41.03.59.1s.33.18.46.33c.13.15.23.34.3.57.07.23.11.5.11.82v.74Zm-.89-1.34c.03.13.04.29.04.48h-.01v.97c0 .19-.01.35-.04.48-.02.13-.06.24-.11.32-.05.08-.12.14-.19.17a.655.655 0 0 1-.5 0 .389.389 0 0 1-.19-.17c-.05-.08-.09-.19-.12-.32s-.04-.29-.04-.48v-.97c0-.19.01-.35.04-.48s.07-.23.12-.31c.05-.08.12-.14.19-.17a.655.655 0 0 1 .5 0c.08.03.14.09.19.17.05.08.09.18.12.31Z"
};
var replay_10 = {
  name: "replay_10",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2v4c4.42 0 8 3.58 8 8s-3.58 8-8 8-8-3.58-8-8h2c0 3.31 2.69 6 6 6s6-2.69 6-6-2.69-6-6-6v4L7 7l5-5Zm-1.95 15h.85v-4.27h-.09l-1.77.63v.69l1.01-.31V17Zm5.13-1.76c0 .32-.03.6-.1.82-.07.22-.17.42-.29.57-.12.15-.28.26-.45.33-.17.07-.37.1-.59.1-.22 0-.41-.03-.59-.1s-.33-.18-.46-.33c-.13-.15-.23-.34-.3-.57-.07-.23-.11-.5-.11-.82v-.74c0-.32.03-.6.1-.82.07-.22.17-.42.29-.57.12-.15.28-.26.45-.33.17-.07.37-.1.59-.1.22 0 .41.03.59.1s.33.18.46.33c.13.15.23.34.3.57.07.23.11.5.11.82v.74Zm-.89-1.34c.03.13.04.29.04.48h-.01v.97c0 .19-.02.35-.04.48s-.06.24-.11.32c-.05.08-.12.14-.19.17a.655.655 0 0 1-.5 0 .389.389 0 0 1-.19-.17c-.05-.08-.09-.19-.12-.32s-.04-.29-.04-.48v-.97c0-.19.01-.35.04-.48s.07-.23.12-.31c.05-.08.12-.14.19-.17a.655.655 0 0 1 .5 0c.08.03.14.09.19.17.05.08.09.18.12.31Z"
};
var replay_5 = {
  name: "replay_5",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2v4c4.42 0 8 3.58 8 8s-3.58 8-8 8-8-3.58-8-8h2c0 3.31 2.69 6 6 6s6-2.69 6-6-2.69-6-6-6v4L7 7l5-5Zm-1.06 10.73-.25 2.17.67.16.022-.023a.29.29 0 0 1 .078-.067c.02-.01.04-.023.06-.035a.66.66 0 0 1 .21-.085c.05-.01.12-.02.2-.02.11 0 .22.02.3.05.08.03.15.08.21.15.06.07.1.14.13.24.03.1.04.2.04.31 0 .11 0 .21-.03.31s-.07.18-.11.25a.49.49 0 0 1-.45.23.65.65 0 0 1-.42-.15c-.11-.09-.17-.23-.19-.41h-.84c0 .2.05.38.13.53.08.15.18.29.32.39.14.1.29.19.46.24.17.05.35.08.53.08.25 0 .46-.05.64-.12.18-.07.33-.18.45-.31s.21-.28.27-.45c.06-.17.09-.35.09-.54 0-.22-.04-.42-.09-.6-.05-.18-.14-.33-.25-.45-.11-.12-.25-.21-.41-.28a1.35 1.35 0 0 0-.75-.08c-.03.005-.06.012-.09.02s-.06.015-.09.02c-.06.01-.11.03-.15.05a.619.619 0 0 1-.05.021c-.022.01-.043.017-.06.029l.11-.92h1.7v-.71h-2.39Z"
};
var replay = {
  name: "replay",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 6V2L7 7l5 5V8c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8Z"
};
var play_circle_outlined = {
  name: "play_circle_outlined",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2Zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8Zm4-8-6 4.5v-9l6 4.5Z"
};
var play_circle = {
  name: "play_circle",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10-4.48 10-10 10S2 17.52 2 12Zm14 0-6-4.5v9l6-4.5Z"
};
var play = {
  name: "play",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m17.5 12-11 7V5l11 7Zm-3.73 0L8.5 8.64v6.72L13.77 12Z"
};
var pause_circle_outlined = {
  name: "pause_circle_outlined",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10-4.48 10-10 10S2 17.52 2 12Zm9 4H9V8h2v8Zm1 4c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8Zm3-4h-2V8h2v8Z"
};
var pause_circle = {
  name: "pause_circle",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2Zm-1 14H9V8h2v8Zm2 0h2V8h-2v8Z"
};
var pause = {
  name: "pause",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M10 19H6V5h4v14Zm4 0V5h4v14h-4Z"
};
var stop = {
  name: "stop",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M6 6h12v12H6V6Zm10 10V8H8v8h8Z"
};
var record = {
  name: "record",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4 12c0-4.42 3.58-8 8-8s8 3.58 8 8-3.58 8-8 8-8-3.58-8-8Zm14 0c0-3.31-2.69-6-6-6s-6 2.69-6 6 2.69 6 6 6 6-2.69 6-6Z"
};
var skip_previous = {
  name: "skip_previous",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M6 6h2v12H6V6Zm3.5 6 8.5 6V6l-8.5 6Zm3.47 0L16 14.14V9.86L12.97 12Z"
};
var skip_next = {
  name: "skip_next",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m6 18 8.5-6L6 6v12Zm2-8.14L11.03 12 8 14.14V9.86ZM18 6h-2v12h2V6Z"
};
var fast_rewind = {
  name: "fast_rewind",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m3.25 12 8.5-6v12l-8.5-6Zm17.5-6-8.5 6 8.5 6V6ZM6.72 12l3.03 2.14V9.86L6.72 12Zm9 0 3.03 2.14V9.86L15.72 12Z"
};
var fast_forward = {
  name: "fast_forward",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m11.75 12-8.5 6V6l8.5 6Zm.5 6 8.5-6-8.5-6v12Zm-3.97-6L5.25 9.86v4.28L8.28 12Zm9 0-3.03-2.14v4.28L17.28 12Z"
};
var closed_caption_outlined = {
  name: "closed_caption_outlined",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 4h14c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Zm0 14h14V6H5v12Zm2-3h3c.55 0 1-.45 1-1v-1H9.5v.5h-2v-3h2v.5H11v-1c0-.55-.45-1-1-1H7c-.55 0-1 .45-1 1v4c0 .55.45 1 1 1Zm10 0h-3c-.55 0-1-.45-1-1v-4c0-.55.45-1 1-1h3c.55 0 1 .45 1 1v1h-1.5v-.5h-2v3h2V13H18v1c0 .55-.45 1-1 1Z"
};
var closed_caption_filled = {
  name: "closed_caption_filled",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 4h14c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Zm2 11h3c.55 0 1-.45 1-1v-1H9.5v.5h-2v-3h2v.5H11v-1c0-.55-.45-1-1-1H7c-.55 0-1 .45-1 1v4c0 .55.45 1 1 1Zm7 0h3c.55 0 1-.45 1-1v-1h-1.5v.5h-2v-3h2v.5H18v-1c0-.55-.45-1-1-1h-3c-.55 0-1 .45-1 1v4c0 .55.45 1 1 1Z"
};
var res_4k_outlined = {
  name: "res_4k_outlined",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M19 3H5c-1.1 0-2 .9-2 2v14a2 2 0 0 0 2 2h14c1.1 0 2-.9 2-2V5a2 2 0 0 0-2-2Zm0 2v14H5V5h14Zm-8 10H9.5v-1.5h-3V9H8v3h1.5V9H11v3h1v1.51h-1V15Zm5.2-3 2-3h-1.7l-2 3V9H13v6h1.5v-3l2 3h1.7l-2-3Z"
};
var res_4k_filled = {
  name: "res_4k_filled",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 3h14a2 2 0 0 1 2 2v14c0 1.1-.9 2-2 2H5a2 2 0 0 1-2-2V5c0-1.1.9-2 2-2Zm6 12H9.5v-1.5h-3V9H8v3h1.5V9H11v3h1v1.51h-1V15Zm5.2-3 2-3h-1.7l-2 3V9H13v6h1.5v-3l2 3h1.7l-2-3Z"
};
var record_voice = {
  name: "record_voice",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m18.44 4.13 1.63-1.63c3.91 4.05 3.9 10.11 0 14l-1.63-1.63c2.77-3.18 2.77-7.72 0-10.74ZM13 9.5c0 2.21-1.79 4-4 4s-4-1.79-4-4 1.79-4 4-4 4 1.79 4 4Zm-2 0c0-1.1-.9-2-2-2s-2 .9-2 2 .9 2 2 2 2-.9 2-2Zm-2 6c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4Zm0 2c-2.69 0-5.78 1.28-6 2h12c-.2-.71-3.3-2-6-2Zm6.08-9.95c.84 1.18.84 2.71 0 3.89l1.68 1.69c2.02-2.02 2.02-5.07 0-7.27l-1.68 1.69Z"
};
var record_voice_off = {
  name: "record_voice_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m18.442 4.13 1.63-1.63c3.86 3.99 3.89 9.94.15 13.83l-1.64-1.64c2.62-3.17 2.58-7.59-.14-10.56Zm-3.36 3.42 1.68-1.69c1.98 2.15 2.01 5.11.11 7.13l-1.7-1.7c.74-1.16.71-2.61-.09-3.74Zm-5.65-2.01 3.53 3.53a3.979 3.979 0 0 0-3.53-3.53Zm-6.43-.77 1.41-1.41 16.73 16.73-1.41 1.41-3.02-3.02c.18.32.29.65.29 1.02v2h-16v-2c0-2.66 5.33-4 8-4 1.78 0 4.74.6 6.51 1.78l-4.4-4.4c-.61.39-1.33.62-2.11.62-2.21 0-4-1.79-4-4 0-.78.23-1.5.62-2.11l-2.62-2.62Zm0 14.73c.22-.72 3.31-2 6-2 2.7 0 5.8 1.29 6 2h-12Zm4-10c0 1.1.9 2 2 2 .22 0 .42-.04.62-.11l-2.51-2.51c-.07.2-.11.4-.11.62Z"
};
var stop_circle = {
  name: "stop_circle",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10-4.48 10-10 10S2 17.52 2 12Zm14-4H8v8h8V8Z"
};
var stop_circle_outlined = {
  name: "stop_circle_outlined",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10-4.48 10-10 10S2 17.52 2 12Zm2 0c0 4.41 3.59 8 8 8s8-3.59 8-8-3.59-8-8-8-8 3.59-8 8Zm12-4H8v8h8V8Z"
};
var expand = {
  name: "expand",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M9.563 12 4 6.645 5.713 5 13 12l-7.287 7L4 17.355 9.563 12Zm7 0L11 6.645 12.713 5 20 12l-7.287 7L11 17.355 16.563 12Z"
};
var last_page = {
  name: "last_page",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m10.385 12-4.59-4.59L7.205 6l6 6-6 6-1.41-1.41 4.59-4.59Zm7.82-6h-2v12h2V6Z"
};
var first_page = {
  name: "first_page",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7.795 6h-2v12h2V6Zm5.82 6 4.59 4.59-1.41 1.41-6-6 6-6 1.41 1.41-4.59 4.59Z"
};
var unfold_more = {
  name: "unfold_more",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15.17 9 12 5.83 8.83 9 7.41 7.59 12 3l4.58 4.59L15.17 9Zm-6.34 6L12 18.17 15.17 15l1.42 1.41L12 21l-4.58-4.59L8.83 15Z"
};
var unfold_less = {
  name: "unfold_less",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m15.17 4 1.42 1.41L12 10 7.41 5.41 8.83 4 12 7.17 15.17 4ZM8.83 20l-1.42-1.41L12 14l4.58 4.59L15.17 20 12 16.83 8.83 20Z"
};
var swap_horizontal = {
  name: "swap_horizontal",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17.01 5 21 9l-3.99 4v-3H10V8h7.01V5ZM3 15l3.99-4v3H14v2H6.99v3L3 15Z"
};
var swap_horizontal_circle = {
  name: "swap_horizontal_circle",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10-4.48 10-10 10S2 17.52 2 12Zm2 0c0 4.41 3.59 8 8 8s8-3.59 8-8-3.59-8-8-8-8 3.59-8 8Zm11-5.5V9h-4v2h4v2.5l3.5-3.5L15 6.5ZM5.5 14 9 10.5V13h4v2H9v2.5L5.5 14Z"
};
var swap_vertical = {
  name: "swap_vertical",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M5 6.99 9 3l4 3.99h-3V14H8V6.99H5ZM16 10v7.01h3L15 21l-4-3.99h3V10h2Z"
};
var swap_vertical_circle = {
  name: "swap_vertical_circle",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 12C2 6.48 6.48 2 12 2s10 4.48 10 10-4.48 10-10 10S2 17.52 2 12Zm2 0c0 4.41 3.59 8 8 8s8-3.59 8-8-3.59-8-8-8-8 3.59-8 8Zm2.5-3L10 5.5 13.5 9H11v4H9V9H6.5Zm7.5 9.5 3.5-3.5H15v-4h-2v4h-2.5l3.5 3.5Z"
};
var chevron_down = {
  name: "chevron_down",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m7.41 8.295 4.59 4.58 4.59-4.58L18 9.705l-6 6-6-6 1.41-1.41Z",
  sizes: {
    small: {
      name: "chevron_down_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M5.175 6 9 9.709 12.825 6 14 7.142 9 12 4 7.142 5.175 6Z"
    }
  }
};
var chevron_left = {
  name: "chevron_left",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15.705 16.59 11.125 12l4.58-4.59L14.295 6l-6 6 6 6 1.41-1.41Z",
  sizes: {
    small: {
      name: "chevron_left_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M12 12.827 8.291 9.002 12 5.177l-1.142-1.175-4.858 5 4.858 5L12 12.827Z"
    }
  }
};
var chevron_right = {
  name: "chevron_right",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m8.295 16.59 4.58-4.59-4.58-4.59L9.705 6l6 6-6 6-1.41-1.41Z",
  sizes: {
    small: {
      name: "chevron_right_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "m6 12.827 3.709-3.825L6 5.177l1.142-1.175 4.858 5-4.858 5L6 12.827Z"
    }
  }
};
var chevron_up = {
  name: "chevron_up",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m7.41 15.705 4.59-4.58 4.59 4.58 1.41-1.41-6-6-6 6 1.41 1.41Z",
  sizes: {
    small: {
      name: "chevron_up_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "m5.174 12.002 3.825-3.709 3.825 3.709 1.175-1.142-5-4.859-5 4.859 1.175 1.142Z"
    }
  }
};
var arrow_back = {
  name: "arrow_back",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2Z",
  sizes: {
    small: {
      name: "arrow_back_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M15 8H6.5l4-4L9 3 3 9l6 6 1.5-1-4-4H15V8Z"
    }
  }
};
var arrow_down = {
  name: "arrow_down",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m20 12-1.41-1.41L13 16.17V4h-2v12.17l-5.58-5.59L4 12l8 8 8-8Z",
  sizes: {
    small: {
      name: "arrow_down_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M8 3v8.5l-4-4L3 9l6 6 6-6-1-1.5-4 4V3H8Z"
    }
  }
};
var arrow_forward = {
  name: "arrow_forward",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m12 4-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8-8-8Z",
  sizes: {
    small: {
      name: "arrow_forward_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M3 10h8.5l-4 4L9 15l6-6-6-6-1.5 1 4 4H3v2Z"
    }
  }
};
var arrow_up = {
  name: "arrow_up",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m4 12 1.41 1.41L11 7.83V20h2V7.83l5.58 5.59L20 12l-8-8-8 8Z",
  sizes: {
    small: {
      name: "arrow_up_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M10 15V6.5l4 4L15 9 9 3 3 9l1 1.5 4-4V15h2Z"
    }
  }
};
var arrow_back_ios = {
  name: "arrow_back_ios",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m17.835 3.87-1.78-1.77-9.89 9.9 9.9 9.9 1.77-1.77L9.705 12l8.13-8.13Z",
  sizes: {
    small: {
      name: "arrow_back_ios_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M13.377 2.903 12.04 1.575 4.624 9l7.425 7.425 1.328-1.327L7.279 9l6.098-6.097Z"
    }
  }
};
var arrow_forward_ios = {
  name: "arrow_forward_ios",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m6.165 20.13 1.77 1.77 9.9-9.9-9.9-9.9-1.77 1.77 8.13 8.13-8.13 8.13Z",
  sizes: {
    small: {
      name: "arrow_forward_ios_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "m4.623 2.903 1.335-1.328L13.376 9l-7.425 7.425-1.328-1.327L10.721 9 4.623 2.903Z"
    }
  }
};
var arrow_drop_down = {
  name: "arrow_drop_down",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m7 9.5 5 5 5-5H7Z",
  sizes: {
    small: {
      name: "arrow_drop_down_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "m4 6.5 5 5 5-5H4Z"
    }
  }
};
var arrow_drop_up = {
  name: "arrow_drop_up",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m7 14.5 5-5 5 5H7Z",
  sizes: {
    small: {
      name: "arrow_drop_up_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "m14 11.5-5-5-5 5h10Z"
    }
  }
};
var arrow_drop_right = {
  name: "arrow_drop_right",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m9.5 17 5-5-5-5v10Z",
  sizes: {
    small: {
      name: "arrow_drop_right_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "m6.5 14 5-5-5-5v10Z"
    }
  }
};
var arrow_drop_left = {
  name: "arrow_drop_left",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m14.5 7-5 5 5 5V7Z",
  sizes: {
    small: {
      name: "arrow_drop_left_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "m11.5 4-5 5 5 5V4Z"
    }
  }
};
var subdirectory_arrow_left = {
  name: "subdirectory_arrow_left",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m10.5 8.5 1.42 1.42-3.59 3.58h9.17v-10h2v12H8.33l3.59 3.58-1.42 1.42-6-6 6-6Z",
  sizes: {
    small: {
      name: "subdirectory_arrow_left_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "m8 7 1 1-2.753 2.125h6.878V3h1.5v8.625H6.247L9 14l-1 1-4.625-4.125L8 7Z"
    }
  }
};
var subdirectory_arrow_right = {
  name: "subdirectory_arrow_right",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m19.5 14.5-6 6-1.42-1.42 3.59-3.58H4.5v-12h2v10h9.17l-3.59-3.58L13.5 8.5l6 6Z",
  sizes: {
    small: {
      name: "subdirectory_arrow_right_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M10 7 9 8l2.752 2.125H4.875V3h-1.5v8.625h8.377L9 14l1 1 4.625-4.125L10 7Z"
    }
  }
};
var collapse = {
  name: "collapse",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7.437 12 13 17.355 11.287 19 4 12l7.287-7L13 6.645 7.437 12Zm7 0L20 17.355 18.287 19 11 12l7.288-7L20 6.645 14.437 12Z"
};
var label = {
  name: "label",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M15.5 5c.67 0 1.27.33 1.63.84L21.5 12l-4.37 6.16c-.36.51-.96.84-1.63.84l-11-.01c-1.1 0-2-.89-2-1.99V7c0-1.1.9-1.99 2-1.99l11-.01Zm-11 12h11l3.55-5-3.55-5h-11v10Z"
};
var label_off = {
  name: "label_off",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 4.22 3.58 5.8c-.36.35-.58.85-.58 1.39v10c0 1.1.9 1.99 2 1.99l11 .01c.28 0 .55-.07.79-.18l2.18 2.18 1.41-1.41L3.41 2.81 2 4.22Zm14 2.97 3.55 5-1.63 2.29 1.43 1.43L22 12.19l-4.37-6.16c-.36-.51-.96-.84-1.63-.84l-7.37.01 2 1.99H16Zm-11 10h9.97L5 7.22v9.97Z"
};
var tag = {
  name: "tag",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m21.41 11.58-9-9C12.05 2.22 11.55 2 11 2H4c-1.1 0-2 .9-2 2v7c0 .55.22 1.05.59 1.42l9 9c.36.36.86.58 1.41.58.55 0 1.05-.22 1.41-.59l7-7c.37-.36.59-.86.59-1.41 0-.55-.23-1.06-.59-1.42ZM13 20.01 4 11V4h7v-.01l9 9-7 7.02ZM5 6.5a1.5 1.5 0 1 1 3 0 1.5 1.5 0 0 1-3 0Z"
};
var high_priority = {
  name: "high_priority",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14 3h-4v12h4V3Zm-4 16a2 2 0 1 1 4 0 2 2 0 0 1-4 0Z"
};
var new_label = {
  name: "new_label",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7.25 12.5 4.75 9H3.5v6h1.25v-3.5L7.3 15h1.2V9H7.25v3.5ZM9.5 15h4v-1.25H11v-1.11h2.5v-1.26H11v-1.12h2.5V9h-4v6Zm9.75-1.5V9h1.25v5c0 .55-.45 1-1 1h-4c-.55 0-1-.45-1-1V9h1.25v4.51h1.13V9.99h1.25v3.51h1.12Z"
};
var new_alert = {
  name: "new_alert",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m20.56 9.215 2.44 2.78-2.44 2.78.34 3.68-3.61.82-1.89 3.18-3.4-1.46-3.4 1.47-1.89-3.18-3.61-.82.34-3.69L1 11.995l2.44-2.79-.34-3.68 3.61-.81 1.89-3.18 3.4 1.46 3.4-1.46 1.89 3.18 3.61.82-.34 3.68Zm-1.81 7.68-.26-2.79 1.85-2.11-1.85-2.11.26-2.79-2.74-.62-1.43-2.41L12 5.175l-2.58-1.1-1.43 2.41-2.74.61.26 2.78-1.85 2.12 1.85 2.1-.26 2.8 2.74.62 1.43 2.41 2.58-1.11 2.58 1.11 1.43-2.41 2.74-.62Zm-5.75-1.9v2h-2v-2h2Zm0-8h-2v6h2v-6Z"
};
var flag = {
  name: "flag",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M4.5 3.5h9l.4 2h5.6v10h-7l-.4-2H6.5v7h-2v-17Zm7.76 4-.4-2H6.5v6h7.24l.4 2h3.36v-6h-5.24Z"
};
var pin = {
  name: "pin",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7.5 9H4v6h1.5v-2h2c.85 0 1.5-.65 1.5-1.5v-1C9 9.65 8.35 9 7.5 9Zm5 6H11V9h1.5v6Zm6.25-2.5V9H20v6h-1.2l-2.55-3.5V15H15V9h1.25l2.5 3.5Zm-13.25-1h2v-1h-2v1Z"
};
var tag_main_equipment = {
  name: "tag_main_equipment",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m12.4 2.6 9 9c.4.3.6.9.6 1.4 0 .5-.2 1-.6 1.4l-7 7c-.3.4-.8.6-1.4.6-.6 0-1.1-.2-1.4-.6l-9-9c-.4-.3-.6-.8-.6-1.4V4c0-1.1.9-2 2-2h7c.6 0 1.1.2 1.4.6ZM4 11l9 9 7-7-9-9H4v7Zm2.5-6C5.7 5 5 5.7 5 6.5S5.7 8 6.5 8 8 7.3 8 6.5 7.3 5 6.5 5Zm10.3 8.7 1.4-1.4-5.8-5.7-1.5 1.5L12 12l-3.8-1.2-1.5 1.5 5.7 5.7 1.4-1.4-1.5-1.5-1.8-1.6 3 .7.9-.9-.8-3 1.7 1.9 1.5 1.5Z"
};
var tag_special_equipment = {
  name: "tag_special_equipment",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m21.4 11.6-9-9c-.3-.4-.8-.6-1.4-.6H4c-1.1 0-2 .9-2 2v7c0 .6.2 1.1.6 1.4l9 9c.3.4.8.6 1.4.6.6 0 1.1-.2 1.4-.6l7-7c.4-.4.6-.9.6-1.4 0-.5-.2-1.1-.6-1.4ZM13 20l-9-9V4h7l9 9-7 7ZM5 6.5C5 5.7 5.7 5 6.5 5S8 5.7 8 6.5 7.3 8 6.5 8 5 7.3 5 6.5Zm6.4 3.8c-.6.3-1.4.7-1.8 1.2-.4.4-.6.8-.3 1.1.361.361.9.126 1.545-.155.961-.42 2.158-.942 3.355.255 1.3 1.3.8 3-.4 4.1-.6.6-1.3 1-2 1.3l-1.4-1.4c.7-.2 1.6-.7 2.1-1.2.4-.4.6-.9.2-1.2-.414-.414-.983-.16-1.649.14-.942.422-2.08.932-3.25-.24-1.3-1.3-.7-2.8.4-3.9.5-.5 1.2-1 1.8-1.2l1.4 1.2Zm6.8 1.8-5.7-5.7L11 7.8l5.7 5.8 1.5-1.5Z"
};
var tag_more = {
  name: "tag_more",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m21.41 11.58-9-9C12.05 2.22 11.55 2 11 2H4c-1.1 0-2 .9-2 2v7c0 .55.22 1.05.59 1.42l9 9c.36.36.86.58 1.41.58.55 0 1.05-.22 1.41-.59l7-7c.37-.36.59-.86.59-1.41 0-.55-.23-1.06-.59-1.42ZM13 20.01 4 11V4h7v-.01l9 9-7 7.02ZM6.657 6.657a1.358 1.358 0 0 1 1.914 0 1.358 1.358 0 0 1 0 1.914 1.358 1.358 0 0 1-1.914 0 1.358 1.358 0 0 1 0-1.914ZM12.4 12.4a1.358 1.358 0 0 1 1.913 0 1.358 1.358 0 0 1 0 1.914 1.358 1.358 0 0 1-1.914 0 1.358 1.358 0 0 1 0-1.914ZM11.442 9.528a1.357 1.357 0 0 0-1.914 0 1.358 1.358 0 0 0 0 1.915 1.358 1.358 0 0 0 1.914 0 1.358 1.358 0 0 0 0-1.915Z"
};
var tag_relations = {
  name: "tag_relations",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M10 4h4v4h-1v3h8v5h1v4h-4v-4h1v-3h-6v3h1v4h-4v-4h1v-3H5v3h1v4H2v-4h1v-5h8V8h-1V4Z"
};
var pregnant_woman = {
  name: "pregnant_woman",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M11 2c-1.11 0-2 .89-2 2 0 1.11.89 2 2 2 1.11 0 2-.89 2-2 0-1.11-.89-2-2-2Zm3 8c1.17.49 1.99 1.66 2 3v4h-3v5h-3v-5H8v-7c0-1.66 1.34-3 3-3s3 1.34 3 3Z"
};
var wheelchair = {
  name: "wheelchair",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M12 2a2 2 0 1 0 0 4 2 2 0 0 0 0-4Zm7 11v-2c-1.54.02-3.09-.75-4.07-1.83l-1.29-1.43c-.17-.19-.38-.34-.61-.45-.005 0-.007-.003-.01-.005-.002-.003-.005-.005-.01-.005H13c-.35-.2-.75-.3-1.19-.26C10.76 7.11 10 8.04 10 9.09V15c0 1.1.9 2 2 2h5v5h2v-5.5c0-1.1-.9-2-2-2h-3v-3.45c1.29 1.07 3.25 1.94 5 1.95ZM7 17c0 1.66 1.34 3 3 3 1.31 0 2.42-.84 2.83-2h2.07A5 5 0 1 1 9 12.1v2.07c-1.16.42-2 1.52-2 2.83Z"
};
var accessible_forward = {
  name: "accessible_forward",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M17.5 2.27a2 2 0 1 0 0 4 2 2 0 0 0 0-4Zm-3 14.46h-2c0 1.65-1.35 3-3 3s-3-1.35-3-3 1.35-3 3-3v-2c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5Zm1.14-3.5h1.86c1.1 0 2 .9 2 2v5.5h-2v-5h-4.98c-1.46 0-2.45-1.57-1.85-2.9l1.83-4.1h-2.21l-.65 1.53-1.92-.53.67-1.8c.33-.73 1.06-1.2 1.87-1.2h5.2c1.48 0 2.46 1.5 1.85 2.83l-1.67 3.67Z"
};
var language = {
  name: "language",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M2 12C2 6.48 6.47 2 11.99 2 17.52 2 22 6.48 22 12s-4.48 10-10.01 10C6.47 22 2 17.52 2 12Zm13.97-4h2.95a8.03 8.03 0 0 0-4.33-3.56c.6 1.11 1.06 2.31 1.38 3.56ZM12 4.04c.83 1.2 1.48 2.53 1.91 3.96h-3.82c.43-1.43 1.08-2.76 1.91-3.96ZM4 12c0 .69.1 1.36.26 2h3.38c-.08-.66-.14-1.32-.14-2 0-.68.06-1.34.14-2H4.26c-.16.64-.26 1.31-.26 2Zm1.08 4h2.95c.32 1.25.78 2.45 1.38 3.56A7.987 7.987 0 0 1 5.08 16Zm0-8h2.95c.32-1.25.78-2.45 1.38-3.56-1.84.63-3.37 1.9-4.33 3.56ZM12 19.96c-.83-1.2-1.48-2.53-1.91-3.96h3.82c-.43 1.43-1.08 2.76-1.91 3.96ZM9.5 12c0 .68.07 1.34.16 2h4.68c.09-.66.16-1.32.16-2 0-.68-.07-1.35-.16-2H9.66c-.09.65-.16 1.32-.16 2Zm5.09 7.56c.6-1.11 1.06-2.31 1.38-3.56h2.95a8.03 8.03 0 0 1-4.33 3.56ZM16.5 12c0 .68-.06 1.34-.14 2h3.38c.16-.64.26-1.31.26-2s-.1-1.36-.26-2h-3.38c.08.66.14 1.32.14 2Z"
};
var google_translate = {
  name: "google_translate",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M20 5h-9.12L10 2H4c-1.1 0-2 .9-2 2v13c0 1.1.9 2 2 2h7l1 3h8c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2ZM7.17 14.59c-2.25 0-4.09-1.83-4.09-4.09s1.83-4.09 4.09-4.09c1.04 0 1.99.37 2.74 1.07l.07.06-1.23 1.18-.06-.05c-.29-.27-.78-.59-1.52-.59-1.31 0-2.38 1.09-2.38 2.42 0 1.33 1.07 2.42 2.38 2.42 1.37 0 1.96-.87 2.12-1.46H7.08V9.91h3.95l.01.07c.04.21.05.4.05.61 0 2.35-1.61 4-3.92 4Zm7.22-.01c-.45-.52-.86-1.1-1.19-1.7l.65 2.23.54-.53Zm-.42-2.46h-.99l-.31-1.04h3.99s-.34 1.31-1.56 2.74c-.52-.62-.89-1.23-1.13-1.7ZM20 21c.55 0 1-.45 1-1V7c0-.55-.45-1-1-1h-8.82l1.18 4.04h1.96V9h1.04v1.04H19v1.04h-1.27c-.32 1.26-1.02 2.48-1.92 3.51l2.71 2.68-.73.73-2.68-2.69-.92.92L15 19l-2 2h7Z"
};
var hearing = {
  name: "hearing",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M7.64 3.03 6.22 1.61A10.965 10.965 0 0 0 3 9.39c0 3.04 1.23 5.79 3.22 7.78l1.41-1.41A9.011 9.011 0 0 1 5 9.39C5 6.9 6.01 4.65 7.64 3.03ZM17 20.39c-.29 0-.56-.06-.76-.15-.71-.37-1.21-.88-1.71-2.38-.505-1.546-1.452-2.277-2.365-2.98l-.025-.02-.01-.008c-.787-.607-1.603-1.238-2.31-2.522-.53-.96-.82-2.01-.82-2.94 0-2.8 2.2-5 5-5s5 2.2 5 5h2c0-3.93-3.07-7-7-7s-7 3.07-7 7c0 1.26.38 2.65 1.07 3.9.91 1.65 1.98 2.48 2.85 3.15.81.62 1.39 1.07 1.71 2.05.6 1.82 1.37 2.84 2.73 3.55A3.999 3.999 0 0 0 21 18.39h-2c0 1.1-.9 2-2 2Zm-3-8.5a2.5 2.5 0 0 1 0-5 2.5 2.5 0 0 1 0 5Z"
};
var translate = {
  name: "translate",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "m12.87 15.07-2.54-2.51.03-.03A17.52 17.52 0 0 0 14.07 6H17V4h-7V2H8v2H1v1.99h11.17C11.5 7.92 10.44 9.75 9 11.35 8.07 10.32 7.3 9.19 6.69 8h-2c.73 1.63 1.73 3.17 2.98 4.56l-5.09 5.02L4 19l5-5 3.11 3.11.76-2.04ZM18.5 10h-2L12 22h2l1.12-3h4.75L21 22h2l-4.5-12Zm-1 2.67L15.88 17h3.24l-1.62-4.33Z"
};
var accessible = {
  name: "accessible",
  prefix: "eds",
  height: "24",
  width: "24",
  svgPathData: "M14 4c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2Zm-2 3c2.83 0 5.89-.3 8.5-1l.5 2c-1.86.5-4 .83-6 1v13h-2v-6h-2v6H9V9c-2-.17-4.14-.5-6-1l.5-2c2.61.7 5.67 1 8.5 1Z",
  sizes: {
    small: {
      name: "accessible_small",
      prefix: "eds",
      height: "18",
      width: "18",
      svgPathData: "M10.5 3c0 .825-.675 1.5-1.5 1.5S7.5 3.825 7.5 3 8.175 1.5 9 1.5s1.5.675 1.5 1.5ZM9 5.25c2.123 0 4.418-.225 6.375-.75L15.75 6c-1.395.375-3 .622-4.5.75v9.75h-1.5V12h-1.5v4.5h-1.5V6.75c-1.5-.128-3.105-.375-4.5-.75l.375-1.5c1.957.525 4.253.75 6.375.75Z"
    }
  }
};
const icons = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  accessible,
  accessible_forward,
  account_circle,
  add,
  add_box,
  add_circle_filled,
  add_circle_outlined,
  aerial_drone,
  alarm,
  alarm_add,
  alarm_off,
  alarm_on,
  all_out,
  anchor,
  android,
  apple_airplay,
  apple_app_store,
  apple_logo,
  approve,
  apps,
  archive,
  arrow_back,
  arrow_back_ios,
  arrow_down,
  arrow_drop_down,
  arrow_drop_left,
  arrow_drop_right,
  arrow_drop_up,
  arrow_forward,
  arrow_forward_ios,
  arrow_up,
  assignment,
  assignment_important,
  assignment_return,
  assignment_returned,
  assignment_turned_in,
  assignment_user,
  attach_file,
  attachment,
  autorenew,
  baby,
  badge,
  bandage,
  bar_chart,
  battery,
  battery_alert,
  battery_charging,
  battery_unknown,
  beach,
  bearing,
  beat,
  bike,
  block,
  blocked,
  blocked_off,
  bluetooth,
  bluetooth_connected,
  bluetooth_disabled,
  bluetooth_searching,
  boat,
  bookmark_collection,
  bookmark_filled,
  bookmark_outlined,
  bookmarks,
  border_all,
  border_bottom,
  border_clear,
  border_color,
  border_horizontal,
  border_inner,
  border_left,
  border_outer,
  border_right,
  border_style,
  border_top,
  border_vertical,
  boundaries,
  briefcase,
  brush,
  bubble_chart,
  build_wrench,
  bus,
  business,
  cable,
  cafe,
  cake,
  calendar,
  calendar_accept,
  calendar_date_range,
  calendar_event,
  calendar_reject,
  calendar_today,
  call,
  call_add,
  call_end,
  camera,
  camera_add_photo,
  car,
  car_wash,
  category,
  change_history,
  check,
  check_circle_outlined,
  checkbox,
  checkbox_indeterminate,
  checkbox_outline,
  chevron_down,
  chevron_left,
  chevron_right,
  chevron_up,
  chrome,
  cinema,
  circuit,
  city,
  clear,
  close,
  close_circle_outlined,
  closed_caption_filled,
  closed_caption_outlined,
  cloud,
  cloud_done,
  cloud_download,
  cloud_off,
  cloud_upload,
  cocktail,
  code,
  coffee,
  collapse,
  collapse_screen,
  collection_1,
  collection_2,
  collection_3,
  collection_4,
  collection_5,
  collection_6,
  collection_7,
  collection_8,
  collection_9,
  collection_9_plus,
  color_palette,
  comment,
  comment_add,
  comment_chat,
  comment_chat_off,
  comment_discussion,
  comment_important,
  comment_more,
  comment_notes,
  comment_solid,
  commute,
  compare,
  compass_calibration,
  computer,
  contact_email,
  contact_phone,
  contacts,
  convenience_store,
  copy,
  craning,
  credit_card,
  crop,
  crop_rotate,
  cut,
  dashboard,
  delete_forever,
  delete_multiple,
  delete_to_trash,
  departure_board,
  desktop_mac,
  desktop_windows,
  details,
  device_unknown,
  dialpad,
  dice,
  dining,
  directions,
  dns,
  do_not_disturb,
  dock,
  dollar,
  done,
  done_all,
  donut_large,
  donut_outlined,
  download,
  download_done,
  drag_handle,
  drink,
  dropper,
  ducting,
  edit,
  edit_text,
  eject,
  electrical,
  email,
  email_alpha,
  email_draft,
  engineering,
  enlarge,
  error_filled,
  error_outlined,
  ev_station,
  exit_to_app,
  expand,
  expand_screen,
  explore,
  explore_off,
  external_link,
  face,
  facebook,
  facebook_messenger,
  fast_food,
  fast_forward,
  fast_rewind,
  fault,
  favorite_filled,
  favorite_outlined,
  file,
  file_add,
  file_copy,
  file_description,
  filter_alt,
  filter_alt_active,
  filter_alt_off,
  filter_list,
  fingerprint_scanner,
  first_page,
  flag,
  flagged,
  flagged_off,
  flame,
  flare,
  flash_off,
  flash_on,
  flight,
  flight_land,
  flight_takeoff,
  flip,
  flip_to_back,
  flip_to_front,
  flower,
  focus_center,
  folder,
  folder_add,
  folder_favorite,
  folder_open,
  folder_shared,
  format_align_center,
  format_align_justify,
  format_align_left,
  format_align_right,
  format_bold,
  format_clear,
  format_color_fill,
  format_color_reset,
  format_color_text,
  format_highlight,
  format_indent_decrease,
  format_indent_increase,
  format_italics,
  format_line_spacing,
  format_list_bulleted,
  format_list_numbered,
  format_quote,
  format_shape,
  format_size,
  format_strikethrough,
  format_underline,
  formula,
  forward_10,
  forward_30,
  forward_5,
  fridge,
  fullscreen,
  fullscreen_exit,
  functions,
  gamepad,
  gas,
  gas_station,
  gavel,
  gear,
  gesture,
  github,
  go_to,
  google_cast,
  google_cast_connected,
  google_maps,
  google_play,
  google_translate,
  gps_fixed,
  gps_not_fixed,
  gps_off,
  grid_layer,
  grid_layers,
  grid_off,
  grid_on,
  grocery_store,
  group,
  group_add,
  gym,
  hand_radio,
  headset,
  headset_mic,
  hearing,
  heat_trace,
  help,
  help_outline,
  high_priority,
  hill_shading,
  history,
  home,
  hospital,
  hotel,
  hourglass_empty,
  hourglass_full,
  image,
  image_add,
  in_progress,
  inbox,
  infinity,
  info_circle,
  insert_link,
  inspect_3d,
  inspect_rotation,
  instagram,
  instrument,
  invert,
  invert_colors,
  ios_logo,
  iphone,
  junction_box,
  key,
  keyboard,
  keyboard_backspace,
  keyboard_capslock,
  keyboard_hide,
  keyboard_return,
  keyboard_space_bar,
  keyboard_tab,
  keyboard_voice,
  label,
  label_off,
  language,
  last_page,
  launch,
  laundry,
  layers,
  layers_off,
  library,
  library_add,
  library_books,
  library_image,
  library_music,
  library_pdf,
  library_video,
  light,
  lightbulb,
  line,
  link,
  link_off,
  linkedin,
  list,
  lock,
  lock_add,
  lock_off,
  lock_open,
  log_in,
  log_out,
  loop,
  mail_unread,
  mall,
  manual_valve,
  map,
  maximize,
  measure,
  meeting_room,
  meeting_room_off,
  memory,
  menu,
  mic,
  mic_off,
  mic_outlined,
  microsoft_edge,
  microsoft_excel,
  microsoft_onedrive,
  microsoft_outlook,
  microsoft_powerpoint,
  microsoft_word,
  minimize,
  miniplayer,
  miniplayer_fullscreen,
  missed_video_call,
  money,
  mood_extremely_happy,
  mood_extremely_sad,
  mood_happy,
  mood_neutral,
  mood_sad,
  mood_very_happy,
  mood_very_sad,
  more_horizontal,
  more_vertical,
  motorcycle,
  mouse,
  move_to_inbox,
  movie,
  movie_file,
  multiline_chart,
  music_note,
  music_note_off,
  nature,
  nature_people,
  navigation,
  near_me,
  new_alert,
  new_label,
  no_craning,
  notifications,
  notifications_active,
  notifications_add,
  notifications_important,
  notifications_off,
  notifications_paused,
  offline,
  offline_document,
  offline_saved,
  oil,
  oil_barrel,
  onshore_drone,
  opacity,
  open_in_browser,
  open_side_sheet,
  pan_tool,
  parking,
  paste,
  pause,
  pause_circle,
  pause_circle_outlined,
  person,
  person_add,
  pharmacy,
  phone,
  pie_chart,
  pin,
  pin_drop,
  pipe_support,
  pizza,
  place,
  place_add,
  place_edit,
  place_person,
  place_unknown,
  placeholder_icon,
  platform,
  play,
  play_circle,
  play_circle_outlined,
  playlist_add,
  playlist_added,
  playlist_play,
  pool,
  power,
  power_bi,
  power_button,
  power_button_off,
  pram,
  pregnant_woman,
  pressure,
  print,
  print_off,
  priority_high,
  priority_low,
  puzzle,
  puzzle_filled,
  radio_button_selected,
  radio_button_unselected,
  railway,
  receipt,
  record,
  record_voice,
  record_voice_off,
  redo,
  reduce,
  refresh,
  remove,
  remove_outlined,
  reorder,
  repeat,
  repeat_one,
  replay,
  replay_10,
  replay_30,
  replay_5,
  reply,
  reply_all,
  report,
  report_bug,
  report_off,
  res_4k_filled,
  res_4k_outlined,
  res_hd_filled,
  res_hd_outlined,
  restaurant,
  restore,
  restore_from_trash,
  restore_page,
  rotate_3d,
  rotate_90_degrees_ccw,
  rotate_left,
  rotate_right,
  router,
  run,
  satellite,
  save,
  scanner,
  scatter_plot,
  school,
  search,
  search_find_replace,
  search_in_page,
  searched_history,
  security,
  select_all,
  send,
  setting_backup_restore,
  settings,
  share,
  share_screen,
  share_screen_off,
  sheet_bottom_position,
  sheet_leftposition,
  sheet_rightposition,
  sheet_topposition,
  shipping,
  shopping_basket,
  shopping_card,
  shopping_cart_add,
  shopping_cart_off,
  shuffle,
  signature,
  sim_card,
  skip_next,
  skip_previous,
  skype,
  slack,
  slideshow,
  smartwatch,
  smoking,
  smoking_off,
  snooze,
  snow,
  sort,
  sort_by_alpha,
  speaker,
  speaker_group,
  spellcheck,
  spotify,
  star_circle,
  star_filled,
  star_half,
  star_outlined,
  stop,
  stop_circle,
  stop_circle_outlined,
  store,
  style: style$1,
  subdirectory_arrow_left,
  subdirectory_arrow_right,
  subsea_drone,
  substation_offshore,
  substation_onshore,
  subway,
  subway_tunnel,
  sun,
  support,
  surface_layer,
  swap_horizontal,
  swap_horizontal_circle,
  swap_vertical,
  swap_vertical_circle,
  switch_off,
  switch_on,
  sync,
  sync_off,
  sync_problem,
  table_chart,
  tablet_android,
  tablet_ipad,
  tag,
  tag_main_equipment,
  tag_more,
  tag_relations,
  tag_special_equipment,
  taxi,
  telecom,
  terrain,
  text_field,
  text_rotation_angled_down,
  text_rotation_angled_up,
  text_rotation_down,
  text_rotation_none,
  text_rotation_up,
  text_rotation_vertical,
  texture,
  thermostat,
  thumb_pin,
  thumbs_down,
  thumbs_up,
  thumbs_up_down,
  ticket,
  time,
  timeline,
  timer,
  timer_off,
  title,
  toc,
  toilet,
  toll,
  toolbox,
  toolbox_rope,
  toolbox_wheel,
  touch,
  track_changes,
  traffic_light,
  train,
  tram,
  transfer,
  transit,
  transit_enter_exit,
  translate,
  trending_down,
  trending_flat,
  trending_up,
  trip_origin,
  tune,
  turbine,
  tv,
  twitter,
  unarchive,
  undo,
  unfold_less,
  unfold_more,
  unsubscribe,
  update,
  upload,
  usb,
  users_circle,
  van,
  verified,
  verified_user,
  vertical_align_bottom,
  vertical_align_center,
  vertical_align_top,
  vertical_split,
  video_call,
  video_chat,
  videocam,
  videocam_off,
  view_360,
  view_agenda,
  view_array,
  view_carousel,
  view_column,
  view_day,
  view_list,
  view_module,
  view_quilt,
  view_stream,
  view_week,
  visibility,
  visibility_off,
  volume_down,
  volume_mute,
  volume_off,
  volume_up,
  walk,
  warning_filled,
  warning_outlined,
  waves,
  well,
  wellbore,
  whats_app,
  wheelchair,
  widgets,
  wifi,
  wifi_off,
  wind_turbine,
  work,
  work_off,
  work_outline,
  world,
  wrap_text,
  youtube,
  youtube_alt,
  zip_file,
  zoom_in,
  zoom_out
}, Symbol.toStringTag, { value: "Module" }));
function get_each_context$3(ctx, list2, i2) {
  const child_ctx = ctx.slice();
  child_ctx[5] = list2[i2];
  return child_ctx;
}
function create_else_block$1(ctx) {
  let path_1;
  return {
    c() {
      path_1 = svg_element("path");
      attr(
        path_1,
        "d",
        /*path*/
        ctx[3]
      );
      attr(
        path_1,
        "fill",
        /*color*/
        ctx[0]
      );
    },
    m(target, anchor2) {
      insert(target, path_1, anchor2);
    },
    p(ctx2, dirty) {
      if (dirty & /*color*/
      1) {
        attr(
          path_1,
          "fill",
          /*color*/
          ctx2[0]
        );
      }
    },
    d(detaching) {
      if (detaching) {
        detach(path_1);
      }
    }
  };
}
function create_if_block$3(ctx) {
  let each_1_anchor;
  let each_value = ensure_array_like(
    /*path*/
    ctx[3]
  );
  let each_blocks = [];
  for (let i2 = 0; i2 < each_value.length; i2 += 1) {
    each_blocks[i2] = create_each_block$3(get_each_context$3(ctx, each_value, i2));
  }
  return {
    c() {
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        each_blocks[i2].c();
      }
      each_1_anchor = empty$1();
    },
    m(target, anchor2) {
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        if (each_blocks[i2]) {
          each_blocks[i2].m(target, anchor2);
        }
      }
      insert(target, each_1_anchor, anchor2);
    },
    p(ctx2, dirty) {
      if (dirty & /*path, color*/
      9) {
        each_value = ensure_array_like(
          /*path*/
          ctx2[3]
        );
        let i2;
        for (i2 = 0; i2 < each_value.length; i2 += 1) {
          const child_ctx = get_each_context$3(ctx2, each_value, i2);
          if (each_blocks[i2]) {
            each_blocks[i2].p(child_ctx, dirty);
          } else {
            each_blocks[i2] = create_each_block$3(child_ctx);
            each_blocks[i2].c();
            each_blocks[i2].m(each_1_anchor.parentNode, each_1_anchor);
          }
        }
        for (; i2 < each_blocks.length; i2 += 1) {
          each_blocks[i2].d(1);
        }
        each_blocks.length = each_value.length;
      }
    },
    d(detaching) {
      if (detaching) {
        detach(each_1_anchor);
      }
      destroy_each(each_blocks, detaching);
    }
  };
}
function create_each_block$3(ctx) {
  let path_1;
  return {
    c() {
      path_1 = svg_element("path");
      attr(
        path_1,
        "d",
        /*p*/
        ctx[5]
      );
      attr(
        path_1,
        "fill",
        /*color*/
        ctx[0]
      );
    },
    m(target, anchor2) {
      insert(target, path_1, anchor2);
    },
    p(ctx2, dirty) {
      if (dirty & /*color*/
      1) {
        attr(
          path_1,
          "fill",
          /*color*/
          ctx2[0]
        );
      }
    },
    d(detaching) {
      if (detaching) {
        detach(path_1);
      }
    }
  };
}
function create_fragment$7(ctx) {
  let svg;
  function select_block_type(ctx2, dirty) {
    if (Array.isArray(
      /*path*/
      ctx2[3]
    ))
      return create_if_block$3;
    return create_else_block$1;
  }
  let current_block_type = select_block_type(ctx);
  let if_block = current_block_type(ctx);
  return {
    c() {
      svg = svg_element("svg");
      if_block.c();
      attr(
        svg,
        "width",
        /*width*/
        ctx[1]
      );
      attr(
        svg,
        "height",
        /*height*/
        ctx[2]
      );
      attr(svg, "viewBox", "0 0 24 24");
      attr(svg, "fill", "none");
      attr(svg, "xmlns", "http://www.w3.org/2000/svg");
    },
    m(target, anchor2) {
      insert(target, svg, anchor2);
      if_block.m(svg, null);
    },
    p(ctx2, [dirty]) {
      if_block.p(ctx2, dirty);
      if (dirty & /*width*/
      2) {
        attr(
          svg,
          "width",
          /*width*/
          ctx2[1]
        );
      }
      if (dirty & /*height*/
      4) {
        attr(
          svg,
          "height",
          /*height*/
          ctx2[2]
        );
      }
    },
    i: noop$2,
    o: noop$2,
    d(detaching) {
      if (detaching) {
        detach(svg);
      }
      if_block.d();
    }
  };
}
function instance$6($$self, $$props, $$invalidate) {
  let { name: name2 = "add" } = $$props;
  let { color: color2 = "currentColor" } = $$props;
  let { width = icons[name2].width } = $$props;
  let { height = icons[name2].height } = $$props;
  const path = icons[name2].svgPathData;
  $$self.$$set = ($$props2) => {
    if ("name" in $$props2)
      $$invalidate(4, name2 = $$props2.name);
    if ("color" in $$props2)
      $$invalidate(0, color2 = $$props2.color);
    if ("width" in $$props2)
      $$invalidate(1, width = $$props2.width);
    if ("height" in $$props2)
      $$invalidate(2, height = $$props2.height);
  };
  return [color2, width, height, path, name2];
}
class Icon extends SvelteComponent {
  constructor(options) {
    super();
    init$1(this, options, instance$6, create_fragment$7, safe_not_equal, { name: 4, color: 0, width: 1, height: 2 });
  }
}
function create_fragment$6(ctx) {
  let div;
  let span;
  let t0;
  let t1;
  let button;
  let icon;
  let current;
  let mounted;
  let dispose;
  icon = new Icon({
    props: { width: "16", height: "16", name: "close" }
  });
  let div_levels = [
    {
      class: "flex items-center gap-1 min-h-7 rounded-full pl-3 chip"
    },
    /*$$restProps*/
    ctx[3]
  ];
  let div_data = {};
  for (let i2 = 0; i2 < div_levels.length; i2 += 1) {
    div_data = assign(div_data, div_levels[i2]);
  }
  return {
    c() {
      div = element("div");
      span = element("span");
      t0 = text(
        /*label*/
        ctx[0]
      );
      t1 = space();
      button = element("button");
      create_component(icon.$$.fragment);
      attr(span, "class", "chip__label svelte-17swhmb");
      attr(button, "aria-label", "Remove");
      attr(button, "class", "flex items-center justify-center rounded-full chip__button svelte-17swhmb");
      set_attributes(div, div_data);
      toggle_class(div, "svelte-17swhmb", true);
    },
    m(target, anchor2) {
      insert(target, div, anchor2);
      append$1(div, span);
      append$1(span, t0);
      append$1(div, t1);
      append$1(div, button);
      mount_component(icon, button, null);
      ctx[4](div);
      current = true;
      if (!mounted) {
        dispose = listen(
          button,
          "click",
          /*onDeleteClick*/
          ctx[2]
        );
        mounted = true;
      }
    },
    p(ctx2, [dirty]) {
      if (!current || dirty & /*label*/
      1)
        set_data(
          t0,
          /*label*/
          ctx2[0]
        );
      set_attributes(div, div_data = get_spread_update(div_levels, [
        {
          class: "flex items-center gap-1 min-h-7 rounded-full pl-3 chip"
        },
        dirty & /*$$restProps*/
        8 && /*$$restProps*/
        ctx2[3]
      ]));
      toggle_class(div, "svelte-17swhmb", true);
    },
    i(local) {
      if (current)
        return;
      transition_in(icon.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(icon.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div);
      }
      destroy_component(icon);
      ctx[4](null);
      mounted = false;
      dispose();
    }
  };
}
function instance$5($$self, $$props, $$invalidate) {
  const omit_props_names = ["label"];
  let $$restProps = compute_rest_props($$props, omit_props_names);
  let { label: label2 = "" } = $$props;
  let chip;
  const dispatch2 = createEventDispatcher();
  const onDeleteClick = () => {
    dispatch2("delete", { chip });
  };
  function div_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      chip = $$value;
      $$invalidate(1, chip);
    });
  }
  $$self.$$set = ($$new_props) => {
    $$props = assign(assign({}, $$props), exclude_internal_props($$new_props));
    $$invalidate(3, $$restProps = compute_rest_props($$props, omit_props_names));
    if ("label" in $$new_props)
      $$invalidate(0, label2 = $$new_props.label);
  };
  return [label2, chip, onDeleteClick, $$restProps, div_binding];
}
class Chip extends SvelteComponent {
  constructor(options) {
    super();
    init$1(this, options, instance$5, create_fragment$6, safe_not_equal, { label: 0 });
  }
}
function get_each_context$2(ctx, list2, i2) {
  const child_ctx = ctx.slice();
  child_ctx[26] = list2[i2];
  child_ctx[29] = i2;
  const constants_0 = (
    /*$option*/
    child_ctx[11](
      /*o*/
      child_ctx[26]
    )
  );
  child_ctx[27] = constants_0;
  return child_ctx;
}
const get_item_slot_changes = (dirty) => ({
  option: dirty[0] & /*options*/
  2,
  index: dirty[0] & /*options*/
  2
});
const get_item_slot_context = (ctx) => ({
  option: (
    /*o*/
    ctx[26]
  ),
  index: (
    /*index*/
    ctx[29]
  )
});
function get_each_context_1(ctx, list2, i2) {
  const child_ctx = ctx.slice();
  child_ctx[30] = list2[i2];
  return child_ctx;
}
function create_else_block_1(ctx) {
  let icon;
  let current;
  icon = new Icon({ props: { name: "arrow_drop_down" } });
  return {
    c() {
      create_component(icon.$$.fragment);
    },
    m(target, anchor2) {
      mount_component(icon, target, anchor2);
      current = true;
    },
    i(local) {
      if (current)
        return;
      transition_in(icon.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(icon.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(icon, detaching);
    }
  };
}
function create_if_block_2$1(ctx) {
  let icon;
  let current;
  icon = new Icon({ props: { name: "arrow_drop_up" } });
  return {
    c() {
      create_component(icon.$$.fragment);
    },
    m(target, anchor2) {
      mount_component(icon, target, anchor2);
      current = true;
    },
    i(local) {
      if (current)
        return;
      transition_in(icon.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(icon.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(icon, detaching);
    }
  };
}
function create_if_block_1$1(ctx) {
  let div;
  let current;
  let each_value_1 = ensure_array_like(
    /*$selected*/
    ctx[5]
  );
  let each_blocks = [];
  for (let i2 = 0; i2 < each_value_1.length; i2 += 1) {
    each_blocks[i2] = create_each_block_1(get_each_context_1(ctx, each_value_1, i2));
  }
  const out = (i2) => transition_out(each_blocks[i2], 1, 1, () => {
    each_blocks[i2] = null;
  });
  return {
    c() {
      div = element("div");
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        each_blocks[i2].c();
      }
      attr(div, "class", "flex gap-1");
    },
    m(target, anchor2) {
      insert(target, div, anchor2);
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        if (each_blocks[i2]) {
          each_blocks[i2].m(div, null);
        }
      }
      current = true;
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*$selected, onChipRemove*/
      524320) {
        each_value_1 = ensure_array_like(
          /*$selected*/
          ctx2[5]
        );
        let i2;
        for (i2 = 0; i2 < each_value_1.length; i2 += 1) {
          const child_ctx = get_each_context_1(ctx2, each_value_1, i2);
          if (each_blocks[i2]) {
            each_blocks[i2].p(child_ctx, dirty);
            transition_in(each_blocks[i2], 1);
          } else {
            each_blocks[i2] = create_each_block_1(child_ctx);
            each_blocks[i2].c();
            transition_in(each_blocks[i2], 1);
            each_blocks[i2].m(div, null);
          }
        }
        group_outros();
        for (i2 = each_value_1.length; i2 < each_blocks.length; i2 += 1) {
          out(i2);
        }
        check_outros();
      }
    },
    i(local) {
      if (current)
        return;
      for (let i2 = 0; i2 < each_value_1.length; i2 += 1) {
        transition_in(each_blocks[i2]);
      }
      current = true;
    },
    o(local) {
      each_blocks = each_blocks.filter(Boolean);
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        transition_out(each_blocks[i2]);
      }
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div);
      }
      destroy_each(each_blocks, detaching);
    }
  };
}
function create_each_block_1(ctx) {
  let chip;
  let current;
  chip = new Chip({
    props: {
      label: (
        /*opt*/
        ctx[30].label
      ),
      "data-value": (
        /*opt*/
        ctx[30].value
      )
    }
  });
  chip.$on(
    "delete",
    /*onChipRemove*/
    ctx[19]
  );
  return {
    c() {
      create_component(chip.$$.fragment);
    },
    m(target, anchor2) {
      mount_component(chip, target, anchor2);
      current = true;
    },
    p(ctx2, dirty) {
      const chip_changes = {};
      if (dirty[0] & /*$selected*/
      32)
        chip_changes.label = /*opt*/
        ctx2[30].label;
      if (dirty[0] & /*$selected*/
      32)
        chip_changes["data-value"] = /*opt*/
        ctx2[30].value;
      chip.$set(chip_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(chip.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(chip.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(chip, detaching);
    }
  };
}
function create_if_block$2(ctx) {
  let ul;
  let each_blocks = [];
  let each_1_lookup = /* @__PURE__ */ new Map();
  let current;
  let mounted;
  let dispose;
  let each_value = ensure_array_like(
    /*options*/
    ctx[1]
  );
  const get_key = (ctx2) => (
    /*index*/
    ctx2[29]
  );
  for (let i2 = 0; i2 < each_value.length; i2 += 1) {
    let child_ctx = get_each_context$2(ctx, each_value, i2);
    let key2 = get_key(child_ctx);
    each_1_lookup.set(key2, each_blocks[i2] = create_each_block$2(key2, child_ctx));
  }
  let each_1_else = null;
  if (!each_value.length) {
    each_1_else = create_else_block();
  }
  let ul_levels = [
    {
      class: "flex flex-col p-2 gap-1 combobox__option-list overflow-y-scroll max-h-[80vh]"
    },
    /*$menu*/
    ctx[10]
  ];
  let ul_data = {};
  for (let i2 = 0; i2 < ul_levels.length; i2 += 1) {
    ul_data = assign(ul_data, ul_levels[i2]);
  }
  return {
    c() {
      ul = element("ul");
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        each_blocks[i2].c();
      }
      if (each_1_else) {
        each_1_else.c();
      }
      set_attributes(ul, ul_data);
      toggle_class(ul, "svelte-1jajbrv", true);
    },
    m(target, anchor2) {
      insert(target, ul, anchor2);
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        if (each_blocks[i2]) {
          each_blocks[i2].m(ul, null);
        }
      }
      if (each_1_else) {
        each_1_else.m(ul, null);
      }
      current = true;
      if (!mounted) {
        dispose = action_destroyer(
          /*$menu*/
          ctx[10].action(ul)
        );
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*$option, options, noMinItemHeight, $$scope*/
      4196370) {
        each_value = ensure_array_like(
          /*options*/
          ctx2[1]
        );
        group_outros();
        each_blocks = update_keyed_each(each_blocks, dirty, get_key, 1, ctx2, each_value, each_1_lookup, ul, outro_and_destroy_block, create_each_block$2, null, get_each_context$2);
        check_outros();
        if (!each_value.length && each_1_else) {
          each_1_else.p(ctx2, dirty);
        } else if (!each_value.length) {
          each_1_else = create_else_block();
          each_1_else.c();
          each_1_else.m(ul, null);
        } else if (each_1_else) {
          each_1_else.d(1);
          each_1_else = null;
        }
      }
      set_attributes(ul, ul_data = get_spread_update(ul_levels, [
        {
          class: "flex flex-col p-2 gap-1 combobox__option-list overflow-y-scroll max-h-[80vh]"
        },
        dirty[0] & /*$menu*/
        1024 && /*$menu*/
        ctx2[10]
      ]));
      toggle_class(ul, "svelte-1jajbrv", true);
    },
    i(local) {
      if (current)
        return;
      for (let i2 = 0; i2 < each_value.length; i2 += 1) {
        transition_in(each_blocks[i2]);
      }
      current = true;
    },
    o(local) {
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        transition_out(each_blocks[i2]);
      }
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(ul);
      }
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        each_blocks[i2].d();
      }
      if (each_1_else)
        each_1_else.d();
      mounted = false;
      dispose();
    }
  };
}
function create_else_block(ctx) {
  let li;
  return {
    c() {
      li = element("li");
      li.textContent = "No results found\n            ";
      attr(li, "class", "relative cursor-pointer rounded-md py-1 pl-8 pr-4");
    },
    m(target, anchor2) {
      insert(target, li, anchor2);
    },
    p: noop$2,
    d(detaching) {
      if (detaching) {
        detach(li);
      }
    }
  };
}
function fallback_block(ctx) {
  let span;
  let t_value = (
    /*o*/
    ctx[26].label + ""
  );
  let t;
  return {
    c() {
      span = element("span");
      t = text(t_value);
    },
    m(target, anchor2) {
      insert(target, span, anchor2);
      append$1(span, t);
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*options*/
      2 && t_value !== (t_value = /*o*/
      ctx2[26].label + ""))
        set_data(t, t_value);
    },
    d(detaching) {
      if (detaching) {
        detach(span);
      }
    }
  };
}
function create_each_block$2(key_1, ctx) {
  let li;
  let t;
  let li_class_value;
  let current;
  let mounted;
  let dispose;
  const item_slot_template = (
    /*#slots*/
    ctx[23].item
  );
  const item_slot = create_slot(
    item_slot_template,
    ctx,
    /*$$scope*/
    ctx[22],
    get_item_slot_context
  );
  const item_slot_or_fallback = item_slot || fallback_block(ctx);
  let li_levels = [
    {
      class: li_class_value = "flex items-center relative px-3 scroll-my-2 combobox__option"
    },
    /*__MELTUI_BUILDER_0__*/
    ctx[27]
  ];
  let li_data = {};
  for (let i2 = 0; i2 < li_levels.length; i2 += 1) {
    li_data = assign(li_data, li_levels[i2]);
  }
  return {
    key: key_1,
    first: null,
    c() {
      li = element("li");
      if (item_slot_or_fallback)
        item_slot_or_fallback.c();
      t = space();
      set_attributes(li, li_data);
      toggle_class(li, "min-h-10", !/*noMinItemHeight*/
      ctx[4]);
      toggle_class(li, "svelte-1jajbrv", true);
      this.first = li;
    },
    m(target, anchor2) {
      insert(target, li, anchor2);
      if (item_slot_or_fallback) {
        item_slot_or_fallback.m(li, null);
      }
      append$1(li, t);
      current = true;
      if (!mounted) {
        dispose = action_destroyer(
          /*__MELTUI_BUILDER_0__*/
          ctx[27].action(li)
        );
        mounted = true;
      }
    },
    p(new_ctx, dirty) {
      ctx = new_ctx;
      if (item_slot) {
        if (item_slot.p && (!current || dirty[0] & /*$$scope, options*/
        4194306)) {
          update_slot_base(
            item_slot,
            item_slot_template,
            ctx,
            /*$$scope*/
            ctx[22],
            !current ? get_all_dirty_from_scope(
              /*$$scope*/
              ctx[22]
            ) : get_slot_changes(
              item_slot_template,
              /*$$scope*/
              ctx[22],
              dirty,
              get_item_slot_changes
            ),
            get_item_slot_context
          );
        }
      } else {
        if (item_slot_or_fallback && item_slot_or_fallback.p && (!current || dirty[0] & /*options*/
        2)) {
          item_slot_or_fallback.p(ctx, !current ? [-1, -1] : dirty);
        }
      }
      set_attributes(li, li_data = get_spread_update(li_levels, [
        { class: li_class_value },
        dirty[0] & /*$option, options*/
        2050 && /*__MELTUI_BUILDER_0__*/
        ctx[27]
      ]));
      toggle_class(li, "min-h-10", !/*noMinItemHeight*/
      ctx[4]);
      toggle_class(li, "svelte-1jajbrv", true);
    },
    i(local) {
      if (current)
        return;
      transition_in(item_slot_or_fallback, local);
      current = true;
    },
    o(local) {
      transition_out(item_slot_or_fallback, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(li);
      }
      if (item_slot_or_fallback)
        item_slot_or_fallback.d(detaching);
      mounted = false;
      dispose();
    }
  };
}
function create_fragment$5(ctx) {
  let div2;
  let label_1;
  let t0;
  let t1;
  let div1;
  let input_1;
  let input_1_value_value;
  let t2;
  let div0;
  let current_block_type_index;
  let if_block0;
  let t3;
  let show_if = (
    /*multiselect*/
    ctx[3] && Array.isArray(
      /*$selected*/
      ctx[5]
    ) && /*$selected*/
    ctx[5].length
  );
  let t4;
  let if_block2_anchor;
  let current;
  let mounted;
  let dispose;
  let label_1_levels = [
    /*$inputLabel*/
    ctx[8],
    { class: "combobox__label" }
  ];
  let label_data = {};
  for (let i2 = 0; i2 < label_1_levels.length; i2 += 1) {
    label_data = assign(label_data, label_1_levels[i2]);
  }
  let input_1_levels = [
    /*$input*/
    ctx[9],
    {
      class: "min-h-10 px-3 w-full combobox__field"
    },
    { placeholder: (
      /*placeholder*/
      ctx[2]
    ) },
    {
      value: input_1_value_value = /*valueLabel*/
      ctx[7]()
    }
  ];
  let input_data = {};
  for (let i2 = 0; i2 < input_1_levels.length; i2 += 1) {
    input_data = assign(input_data, input_1_levels[i2]);
  }
  const if_block_creators = [create_if_block_2$1, create_else_block_1];
  const if_blocks = [];
  function select_block_type(ctx2, dirty) {
    if (
      /*$open*/
      ctx2[6]
    )
      return 0;
    return 1;
  }
  current_block_type_index = select_block_type(ctx);
  if_block0 = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
  let if_block1 = show_if && create_if_block_1$1(ctx);
  let if_block2 = (
    /*$open*/
    ctx[6] && create_if_block$2(ctx)
  );
  return {
    c() {
      div2 = element("div");
      label_1 = element("label");
      t0 = text(
        /*label*/
        ctx[0]
      );
      t1 = space();
      div1 = element("div");
      input_1 = element("input");
      t2 = space();
      div0 = element("div");
      if_block0.c();
      t3 = space();
      if (if_block1)
        if_block1.c();
      t4 = space();
      if (if_block2)
        if_block2.c();
      if_block2_anchor = empty$1();
      set_attributes(label_1, label_data);
      toggle_class(label_1, "svelte-1jajbrv", true);
      set_attributes(input_1, input_data);
      toggle_class(input_1, "svelte-1jajbrv", true);
      attr(div0, "class", "absolute right-2 top-1/2 z-10 -translate-y-1/2");
      attr(div1, "class", "relative");
      attr(div2, "class", "flex flex-col gap-1 combobox");
    },
    m(target, anchor2) {
      insert(target, div2, anchor2);
      append$1(div2, label_1);
      append$1(label_1, t0);
      append$1(div2, t1);
      append$1(div2, div1);
      append$1(div1, input_1);
      if ("value" in input_data) {
        input_1.value = input_data.value;
      }
      if (input_1.autofocus)
        input_1.focus();
      append$1(div1, t2);
      append$1(div1, div0);
      if_blocks[current_block_type_index].m(div0, null);
      append$1(div2, t3);
      if (if_block1)
        if_block1.m(div2, null);
      insert(target, t4, anchor2);
      if (if_block2)
        if_block2.m(target, anchor2);
      insert(target, if_block2_anchor, anchor2);
      current = true;
      if (!mounted) {
        dispose = [
          action_destroyer(
            /*$inputLabel*/
            ctx[8].action(label_1)
          ),
          action_destroyer(
            /*$input*/
            ctx[9].action(input_1)
          )
        ];
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      if (!current || dirty[0] & /*label*/
      1)
        set_data_maybe_contenteditable(
          t0,
          /*label*/
          ctx2[0],
          label_data["contenteditable"]
        );
      set_attributes(label_1, label_data = get_spread_update(label_1_levels, [
        dirty[0] & /*$inputLabel*/
        256 && /*$inputLabel*/
        ctx2[8],
        { class: "combobox__label" }
      ]));
      toggle_class(label_1, "svelte-1jajbrv", true);
      set_attributes(input_1, input_data = get_spread_update(input_1_levels, [
        dirty[0] & /*$input*/
        512 && /*$input*/
        ctx2[9],
        {
          class: "min-h-10 px-3 w-full combobox__field"
        },
        (!current || dirty[0] & /*placeholder*/
        4) && { placeholder: (
          /*placeholder*/
          ctx2[2]
        ) },
        (!current || dirty[0] & /*valueLabel*/
        128 && input_1_value_value !== (input_1_value_value = /*valueLabel*/
        ctx2[7]()) && input_1.value !== input_1_value_value) && { value: input_1_value_value }
      ]));
      if ("value" in input_data) {
        input_1.value = input_data.value;
      }
      toggle_class(input_1, "svelte-1jajbrv", true);
      let previous_block_index = current_block_type_index;
      current_block_type_index = select_block_type(ctx2);
      if (current_block_type_index !== previous_block_index) {
        group_outros();
        transition_out(if_blocks[previous_block_index], 1, 1, () => {
          if_blocks[previous_block_index] = null;
        });
        check_outros();
        if_block0 = if_blocks[current_block_type_index];
        if (!if_block0) {
          if_block0 = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx2);
          if_block0.c();
        }
        transition_in(if_block0, 1);
        if_block0.m(div0, null);
      }
      if (dirty[0] & /*multiselect, $selected*/
      40)
        show_if = /*multiselect*/
        ctx2[3] && Array.isArray(
          /*$selected*/
          ctx2[5]
        ) && /*$selected*/
        ctx2[5].length;
      if (show_if) {
        if (if_block1) {
          if_block1.p(ctx2, dirty);
          if (dirty[0] & /*multiselect, $selected*/
          40) {
            transition_in(if_block1, 1);
          }
        } else {
          if_block1 = create_if_block_1$1(ctx2);
          if_block1.c();
          transition_in(if_block1, 1);
          if_block1.m(div2, null);
        }
      } else if (if_block1) {
        group_outros();
        transition_out(if_block1, 1, 1, () => {
          if_block1 = null;
        });
        check_outros();
      }
      if (
        /*$open*/
        ctx2[6]
      ) {
        if (if_block2) {
          if_block2.p(ctx2, dirty);
          if (dirty[0] & /*$open*/
          64) {
            transition_in(if_block2, 1);
          }
        } else {
          if_block2 = create_if_block$2(ctx2);
          if_block2.c();
          transition_in(if_block2, 1);
          if_block2.m(if_block2_anchor.parentNode, if_block2_anchor);
        }
      } else if (if_block2) {
        group_outros();
        transition_out(if_block2, 1, 1, () => {
          if_block2 = null;
        });
        check_outros();
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block0);
      transition_in(if_block1);
      transition_in(if_block2);
      current = true;
    },
    o(local) {
      transition_out(if_block0);
      transition_out(if_block1);
      transition_out(if_block2);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div2);
        detach(t4);
        detach(if_block2_anchor);
      }
      if_blocks[current_block_type_index].d();
      if (if_block1)
        if_block1.d();
      if (if_block2)
        if_block2.d(detaching);
      mounted = false;
      run_all(dispose);
    }
  };
}
function instance$4($$self, $$props, $$invalidate) {
  let valueLabel;
  let $selected;
  let $inputValue;
  let $open;
  let $inputLabel;
  let $input;
  let $menu;
  let $option;
  let { $$slots: slots = {}, $$scope } = $$props;
  let { label: label2 = "" } = $$props;
  let { options = [] } = $$props;
  let { placeholder = "" } = $$props;
  let { multiselect = false } = $$props;
  let { value } = $$props;
  let { noMinItemHeight = false } = $$props;
  let { customValueLabel } = $$props;
  const dispatch2 = createEventDispatcher();
  const { elements: { menu: menu2, input, option, label: inputLabel }, states: { open, inputValue, selected } } = createCombobox({
    forceVisible: true,
    multiple: multiselect,
    onSelectedChange: ({ next: next2 }) => {
      if (multiselect) {
        $$invalidate(20, value = next2.map((v) => v.value));
      } else {
        $$invalidate(20, value = next2.value);
      }
      dispatch2("change", { value });
      return next2;
    },
    defaultSelected: options.filter((o) => !!value && value.includes(o.value))
  });
  component_subscribe($$self, menu2, (value2) => $$invalidate(10, $menu = value2));
  component_subscribe($$self, input, (value2) => $$invalidate(9, $input = value2));
  component_subscribe($$self, option, (value2) => $$invalidate(11, $option = value2));
  component_subscribe($$self, inputLabel, (value2) => $$invalidate(8, $inputLabel = value2));
  component_subscribe($$self, open, (value2) => $$invalidate(6, $open = value2));
  component_subscribe($$self, inputValue, (value2) => $$invalidate(24, $inputValue = value2));
  component_subscribe($$self, selected, (value2) => $$invalidate(5, $selected = value2));
  const onChipRemove = (e) => {
    const { value: value2 } = e.detail.chip.dataset;
    selected.update((opts) => {
      const idx = opts.findIndex((opt) => opt.value === value2);
      return opts.filter((_, i2) => i2 !== idx);
    });
  };
  $$self.$$set = ($$props2) => {
    if ("label" in $$props2)
      $$invalidate(0, label2 = $$props2.label);
    if ("options" in $$props2)
      $$invalidate(1, options = $$props2.options);
    if ("placeholder" in $$props2)
      $$invalidate(2, placeholder = $$props2.placeholder);
    if ("multiselect" in $$props2)
      $$invalidate(3, multiselect = $$props2.multiselect);
    if ("value" in $$props2)
      $$invalidate(20, value = $$props2.value);
    if ("noMinItemHeight" in $$props2)
      $$invalidate(4, noMinItemHeight = $$props2.noMinItemHeight);
    if ("customValueLabel" in $$props2)
      $$invalidate(21, customValueLabel = $$props2.customValueLabel);
    if ("$$scope" in $$props2)
      $$invalidate(22, $$scope = $$props2.$$scope);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty[0] & /*multiselect, $open, $selected*/
    104) {
      if (!multiselect && !$open) {
        set_store_value(inputValue, $inputValue = ($selected == null ? void 0 : $selected.label) ?? "", $inputValue);
      }
    }
    if ($$self.$$.dirty[0] & /*customValueLabel, value, options, multiselect*/
    3145738) {
      $$invalidate(7, valueLabel = () => {
        if (customValueLabel)
          return customValueLabel(value, options);
        if (multiselect) {
          return value.map((v) => options.find((o) => o.value === v).label);
        } else {
          return options.find((o) => o.value === value).label;
        }
      });
    }
  };
  return [
    label2,
    options,
    placeholder,
    multiselect,
    noMinItemHeight,
    $selected,
    $open,
    valueLabel,
    $inputLabel,
    $input,
    $menu,
    $option,
    menu2,
    input,
    option,
    inputLabel,
    open,
    inputValue,
    selected,
    onChipRemove,
    value,
    customValueLabel,
    $$scope,
    slots
  ];
}
class Combobox extends SvelteComponent {
  constructor(options) {
    super();
    init$1(
      this,
      options,
      instance$4,
      create_fragment$5,
      safe_not_equal,
      {
        label: 0,
        options: 1,
        placeholder: 2,
        multiselect: 3,
        value: 20,
        noMinItemHeight: 4,
        customValueLabel: 21
      },
      null,
      [-1, -1]
    );
  }
}
function create_fragment$4(ctx) {
  let g;
  let path;
  let path_d_value;
  let path_levels = [
    /*style*/
    ctx[0],
    { d: path_d_value = /*compute*/
    ctx[1]() }
  ];
  let path_data = {};
  for (let i2 = 0; i2 < path_levels.length; i2 += 1) {
    path_data = assign(path_data, path_levels[i2]);
  }
  return {
    c() {
      g = svg_element("g");
      path = svg_element("path");
      set_svg_attributes(path, path_data);
    },
    m(target, anchor2) {
      insert(target, g, anchor2);
      append$1(g, path);
    },
    p(ctx2, [dirty]) {
      set_svg_attributes(path, path_data = get_spread_update(path_levels, [
        dirty & /*style*/
        1 && /*style*/
        ctx2[0],
        dirty & /*compute*/
        2 && path_d_value !== (path_d_value = /*compute*/
        ctx2[1]()) && { d: path_d_value }
      ]));
    },
    i: noop$2,
    o: noop$2,
    d(detaching) {
      if (detaching) {
        detach(g);
      }
    }
  };
}
function instance$3($$self, $$props, $$invalidate) {
  let compute;
  let { width } = $$props;
  let { height } = $$props;
  let { domainY } = $$props;
  let { domainX } = $$props;
  let { sharedDomainX } = $$props;
  let { sharedDomainY } = $$props;
  let { points } = $$props;
  let { realization } = $$props;
  let { style: style2 } = $$props;
  $$self.$$set = ($$props2) => {
    if ("width" in $$props2)
      $$invalidate(2, width = $$props2.width);
    if ("height" in $$props2)
      $$invalidate(3, height = $$props2.height);
    if ("domainY" in $$props2)
      $$invalidate(4, domainY = $$props2.domainY);
    if ("domainX" in $$props2)
      $$invalidate(5, domainX = $$props2.domainX);
    if ("sharedDomainX" in $$props2)
      $$invalidate(6, sharedDomainX = $$props2.sharedDomainX);
    if ("sharedDomainY" in $$props2)
      $$invalidate(7, sharedDomainY = $$props2.sharedDomainY);
    if ("points" in $$props2)
      $$invalidate(8, points = $$props2.points);
    if ("realization" in $$props2)
      $$invalidate(9, realization = $$props2.realization);
    if ("style" in $$props2)
      $$invalidate(0, style2 = $$props2.style);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty & /*sharedDomainX, width, sharedDomainY, height, points*/
    460) {
      $$invalidate(1, compute = () => {
        const scaleXNormalizedToShared = linear().domain([0, 1]).range(sharedDomainX);
        const scaleXSharedToViewport = linear().domain(sharedDomainX).range([0, width]);
        const scaleYNormalizedToShared = linear().domain([0, 1]).range(sharedDomainY);
        const scaleYSharedToViewport = linear().domain(sharedDomainY).range([height, 0]);
        const scaleX = (x2) => scaleXSharedToViewport(scaleXNormalizedToShared(x2));
        const scaleY = (y2) => scaleYSharedToViewport(scaleYNormalizedToShared(y2));
        const lineGenerator = area().x((d) => scaleX(d[0])).y1((d) => scaleY(d[1])).y0((d) => scaleY(0)).curve(basis);
        const lineDataString = lineGenerator(points);
        if (!lineDataString) {
          throw new Error("Failed to create line from data");
        }
        return lineDataString;
      });
    }
  };
  return [
    style2,
    compute,
    width,
    height,
    domainY,
    domainX,
    sharedDomainX,
    sharedDomainY,
    points,
    realization
  ];
}
class AreaLayer extends SvelteComponent {
  constructor(options) {
    super();
    init$1(this, options, instance$3, create_fragment$4, safe_not_equal, {
      width: 2,
      height: 3,
      domainY: 4,
      domainX: 5,
      sharedDomainX: 6,
      sharedDomainY: 7,
      points: 8,
      realization: 9,
      style: 0
    });
  }
}
function get_each_context$1(ctx, list2, i2) {
  const child_ctx = ctx.slice();
  child_ctx[14] = list2[i2];
  return child_ctx;
}
function create_if_block$1(ctx) {
  let g;
  let singleaxislayer;
  let g_transform_value;
  let current;
  singleaxislayer = new SingleAxisLayer({
    props: {
      width: (
        /*width*/
        ctx[6] - /*axisMarginLeft*/
        ctx[1]
      ),
      height: (
        /*height*/
        ctx[7] - /*axisMarginBottom*/
        ctx[2]
      ),
      domainX: (
        /*kdeCharts*/
        ctx[10]().sharedDomainX
      ),
      domainY: (
        /*kdeCharts*/
        ctx[10]().sharedDomainY
      ),
      formatY: ".0",
      ticksY: 14,
      ticksX: 15,
      formatX: ".1f"
    }
  });
  return {
    c() {
      g = svg_element("g");
      create_component(singleaxislayer.$$.fragment);
      attr(g, "transform", g_transform_value = `translate(${/*axisMarginLeft*/
      ctx[1]},0)`);
    },
    m(target, anchor2) {
      insert(target, g, anchor2);
      mount_component(singleaxislayer, g, null);
      current = true;
    },
    p(ctx2, dirty) {
      const singleaxislayer_changes = {};
      if (dirty & /*width, axisMarginLeft*/
      66)
        singleaxislayer_changes.width = /*width*/
        ctx2[6] - /*axisMarginLeft*/
        ctx2[1];
      if (dirty & /*height, axisMarginBottom*/
      132)
        singleaxislayer_changes.height = /*height*/
        ctx2[7] - /*axisMarginBottom*/
        ctx2[2];
      if (dirty & /*kdeCharts*/
      1024)
        singleaxislayer_changes.domainX = /*kdeCharts*/
        ctx2[10]().sharedDomainX;
      if (dirty & /*kdeCharts*/
      1024)
        singleaxislayer_changes.domainY = /*kdeCharts*/
        ctx2[10]().sharedDomainY;
      singleaxislayer.$set(singleaxislayer_changes);
      if (!current || dirty & /*axisMarginLeft*/
      2 && g_transform_value !== (g_transform_value = `translate(${/*axisMarginLeft*/
      ctx2[1]},0)`)) {
        attr(g, "transform", g_transform_value);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(singleaxislayer.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(singleaxislayer.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(g);
      }
      destroy_component(singleaxislayer);
    }
  };
}
function create_each_block$1(ctx) {
  let arealayer;
  let current;
  arealayer = new AreaLayer({
    props: {
      width: (
        /*width*/
        ctx[6] - /*axisMarginLeft*/
        ctx[1]
      ),
      height: (
        /*height*/
        ctx[7] - /*axisMarginBottom*/
        ctx[2]
      ),
      domainX: (
        /*chart*/
        ctx[14].domainX
      ),
      domainY: (
        /*chart*/
        ctx[14].domainY
      ),
      sharedDomainX: (
        /*kdeCharts*/
        ctx[10]().sharedDomainX
      ),
      sharedDomainY: (
        /*kdeCharts*/
        ctx[10]().sharedDomainY
      ),
      points: (
        /*chart*/
        ctx[14].points
      ),
      style: (
        /*style*/
        ctx[5][
          /*chart*/
          ctx[14].ensemble
        ]
      ),
      realization: (
        /*chart*/
        ctx[14].ensemble
      )
    }
  });
  return {
    c() {
      create_component(arealayer.$$.fragment);
    },
    m(target, anchor2) {
      mount_component(arealayer, target, anchor2);
      current = true;
    },
    p(ctx2, dirty) {
      const arealayer_changes = {};
      if (dirty & /*width, axisMarginLeft*/
      66)
        arealayer_changes.width = /*width*/
        ctx2[6] - /*axisMarginLeft*/
        ctx2[1];
      if (dirty & /*height, axisMarginBottom*/
      132)
        arealayer_changes.height = /*height*/
        ctx2[7] - /*axisMarginBottom*/
        ctx2[2];
      if (dirty & /*kdeCharts*/
      1024)
        arealayer_changes.domainX = /*chart*/
        ctx2[14].domainX;
      if (dirty & /*kdeCharts*/
      1024)
        arealayer_changes.domainY = /*chart*/
        ctx2[14].domainY;
      if (dirty & /*kdeCharts*/
      1024)
        arealayer_changes.sharedDomainX = /*kdeCharts*/
        ctx2[10]().sharedDomainX;
      if (dirty & /*kdeCharts*/
      1024)
        arealayer_changes.sharedDomainY = /*kdeCharts*/
        ctx2[10]().sharedDomainY;
      if (dirty & /*kdeCharts*/
      1024)
        arealayer_changes.points = /*chart*/
        ctx2[14].points;
      if (dirty & /*style, kdeCharts*/
      1056)
        arealayer_changes.style = /*style*/
        ctx2[5][
          /*chart*/
          ctx2[14].ensemble
        ];
      if (dirty & /*kdeCharts*/
      1024)
        arealayer_changes.realization = /*chart*/
        ctx2[14].ensemble;
      arealayer.$set(arealayer_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(arealayer.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(arealayer.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(arealayer, detaching);
    }
  };
}
function create_fragment$3(ctx) {
  let div;
  let slider0;
  let updating_value;
  let t0;
  let slider1;
  let updating_value_1;
  let portal_action;
  let t1;
  let svg;
  let g;
  let g_transform_value;
  let current;
  let mounted;
  let dispose;
  function slider0_value_binding(value) {
    ctx[12](value);
  }
  let slider0_props = {
    min: 0.01,
    max: 2,
    step: 0.01,
    valueDisplay: func$1
  };
  if (
    /*bandwidth*/
    ctx[8] !== void 0
  ) {
    slider0_props.value = /*bandwidth*/
    ctx[8];
  }
  slider0 = new Slider({ props: slider0_props });
  binding_callbacks.push(() => bind(slider0, "value", slider0_value_binding));
  function slider1_value_binding(value) {
    ctx[13](value);
  }
  let slider1_props = {
    min: 2,
    max: 100,
    step: 1,
    valueDisplay: func_1
  };
  if (
    /*numPoints*/
    ctx[9] !== void 0
  ) {
    slider1_props.value = /*numPoints*/
    ctx[9];
  }
  slider1 = new Slider({ props: slider1_props });
  binding_callbacks.push(() => bind(slider1, "value", slider1_value_binding));
  let if_block = (
    /*showAxisX*/
    (ctx[3] || /*showAxisY*/
    ctx[4]) && create_if_block$1(ctx)
  );
  let each_value = ensure_array_like(
    /*kdeCharts*/
    ctx[10]().lines
  );
  let each_blocks = [];
  for (let i2 = 0; i2 < each_value.length; i2 += 1) {
    each_blocks[i2] = create_each_block$1(get_each_context$1(ctx, each_value, i2));
  }
  const out = (i2) => transition_out(each_blocks[i2], 1, 1, () => {
    each_blocks[i2] = null;
  });
  return {
    c() {
      div = element("div");
      create_component(slider0.$$.fragment);
      t0 = space();
      create_component(slider1.$$.fragment);
      t1 = space();
      svg = svg_element("svg");
      if (if_block)
        if_block.c();
      g = svg_element("g");
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        each_blocks[i2].c();
      }
      attr(g, "transform", g_transform_value = `translate(${/*axisMarginLeft*/
      ctx[1]},0)`);
      attr(svg, "class", "absolute top-0 left-0 h-full w-full overflow-visible");
    },
    m(target, anchor2) {
      insert(target, div, anchor2);
      mount_component(slider0, div, null);
      append$1(div, t0);
      mount_component(slider1, div, null);
      insert(target, t1, anchor2);
      insert(target, svg, anchor2);
      if (if_block)
        if_block.m(svg, null);
      append$1(svg, g);
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        if (each_blocks[i2]) {
          each_blocks[i2].m(g, null);
        }
      }
      current = true;
      if (!mounted) {
        dispose = action_destroyer(portal_action = portal.call(
          null,
          div,
          /*localControlsDivSelector*/
          ctx[0]
        ));
        mounted = true;
      }
    },
    p(ctx2, [dirty]) {
      const slider0_changes = {};
      if (!updating_value && dirty & /*bandwidth*/
      256) {
        updating_value = true;
        slider0_changes.value = /*bandwidth*/
        ctx2[8];
        add_flush_callback(() => updating_value = false);
      }
      slider0.$set(slider0_changes);
      const slider1_changes = {};
      if (!updating_value_1 && dirty & /*numPoints*/
      512) {
        updating_value_1 = true;
        slider1_changes.value = /*numPoints*/
        ctx2[9];
        add_flush_callback(() => updating_value_1 = false);
      }
      slider1.$set(slider1_changes);
      if (portal_action && is_function(portal_action.update) && dirty & /*localControlsDivSelector*/
      1)
        portal_action.update.call(
          null,
          /*localControlsDivSelector*/
          ctx2[0]
        );
      if (
        /*showAxisX*/
        ctx2[3] || /*showAxisY*/
        ctx2[4]
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
          if (dirty & /*showAxisX, showAxisY*/
          24) {
            transition_in(if_block, 1);
          }
        } else {
          if_block = create_if_block$1(ctx2);
          if_block.c();
          transition_in(if_block, 1);
          if_block.m(svg, g);
        }
      } else if (if_block) {
        group_outros();
        transition_out(if_block, 1, 1, () => {
          if_block = null;
        });
        check_outros();
      }
      if (dirty & /*width, axisMarginLeft, height, axisMarginBottom, kdeCharts, style*/
      1254) {
        each_value = ensure_array_like(
          /*kdeCharts*/
          ctx2[10]().lines
        );
        let i2;
        for (i2 = 0; i2 < each_value.length; i2 += 1) {
          const child_ctx = get_each_context$1(ctx2, each_value, i2);
          if (each_blocks[i2]) {
            each_blocks[i2].p(child_ctx, dirty);
            transition_in(each_blocks[i2], 1);
          } else {
            each_blocks[i2] = create_each_block$1(child_ctx);
            each_blocks[i2].c();
            transition_in(each_blocks[i2], 1);
            each_blocks[i2].m(g, null);
          }
        }
        group_outros();
        for (i2 = each_value.length; i2 < each_blocks.length; i2 += 1) {
          out(i2);
        }
        check_outros();
      }
      if (!current || dirty & /*axisMarginLeft*/
      2 && g_transform_value !== (g_transform_value = `translate(${/*axisMarginLeft*/
      ctx2[1]},0)`)) {
        attr(g, "transform", g_transform_value);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(slider0.$$.fragment, local);
      transition_in(slider1.$$.fragment, local);
      transition_in(if_block);
      for (let i2 = 0; i2 < each_value.length; i2 += 1) {
        transition_in(each_blocks[i2]);
      }
      current = true;
    },
    o(local) {
      transition_out(slider0.$$.fragment, local);
      transition_out(slider1.$$.fragment, local);
      transition_out(if_block);
      each_blocks = each_blocks.filter(Boolean);
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        transition_out(each_blocks[i2]);
      }
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div);
        detach(t1);
        detach(svg);
      }
      destroy_component(slider0);
      destroy_component(slider1);
      if (if_block)
        if_block.d();
      destroy_each(each_blocks, detaching);
      mounted = false;
      dispose();
    }
  };
}
const func$1 = (v) => `KDE Bandwidth: ${v}`;
const func_1 = (v) => `Number of points: ${v}`;
function instance$2($$self, $$props, $$invalidate) {
  let kdeCharts;
  let { localControlsDivSelector } = $$props;
  let { axisMarginLeft = 35 } = $$props;
  let { axisMarginBottom = 35 } = $$props;
  let { showAxisX = true } = $$props;
  let { showAxisY = true } = $$props;
  let { data } = $$props;
  let { style: style2 } = $$props;
  let { width } = $$props;
  let { height } = $$props;
  let bandwidth = 1.1;
  let numPoints = 50;
  function slider0_value_binding(value) {
    bandwidth = value;
    $$invalidate(8, bandwidth);
  }
  function slider1_value_binding(value) {
    numPoints = value;
    $$invalidate(9, numPoints);
  }
  $$self.$$set = ($$props2) => {
    if ("localControlsDivSelector" in $$props2)
      $$invalidate(0, localControlsDivSelector = $$props2.localControlsDivSelector);
    if ("axisMarginLeft" in $$props2)
      $$invalidate(1, axisMarginLeft = $$props2.axisMarginLeft);
    if ("axisMarginBottom" in $$props2)
      $$invalidate(2, axisMarginBottom = $$props2.axisMarginBottom);
    if ("showAxisX" in $$props2)
      $$invalidate(3, showAxisX = $$props2.showAxisX);
    if ("showAxisY" in $$props2)
      $$invalidate(4, showAxisY = $$props2.showAxisY);
    if ("data" in $$props2)
      $$invalidate(11, data = $$props2.data);
    if ("style" in $$props2)
      $$invalidate(5, style2 = $$props2.style);
    if ("width" in $$props2)
      $$invalidate(6, width = $$props2.width);
    if ("height" in $$props2)
      $$invalidate(7, height = $$props2.height);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty & /*data, bandwidth, numPoints*/
    2816) {
      $$invalidate(10, kdeCharts = () => {
        const distinctEnsembles = [...new Set(data.map((d) => d.ensemble_id).values()).values()];
        const kdeInfosPerEnsemble = distinctEnsembles.map((ensemble) => ({
          ensemble,
          values: data.filter((d) => d.ensemble_id === ensemble).map((d) => d.values)
        })).map(({ ensemble, values }) => ({
          ensemble,
          info: kde(values, bandwidth, kdeEpanechnikov(0.3), numPoints)
        }));
        const sharedDomainY = [
          Math.min(...kdeInfosPerEnsemble.map((d) => d.info.domainY[0])),
          Math.max(...kdeInfosPerEnsemble.map((d) => d.info.domainY[1]))
        ];
        const sharedDomainX = [
          Math.min(...kdeInfosPerEnsemble.map((d) => d.info.domainX[0])),
          Math.max(...kdeInfosPerEnsemble.map((d) => d.info.domainX[1]))
        ];
        const lines = kdeInfosPerEnsemble.map((d) => ({
          points: d.info.kdeValues,
          domainX: d.info.domainX,
          domainY: d.info.domainY,
          ensemble: d.ensemble
        }));
        return { lines, sharedDomainY, sharedDomainX };
      });
    }
  };
  return [
    localControlsDivSelector,
    axisMarginLeft,
    axisMarginBottom,
    showAxisX,
    showAxisY,
    style2,
    width,
    height,
    bandwidth,
    numPoints,
    kdeCharts,
    data,
    slider0_value_binding,
    slider1_value_binding
  ];
}
class ParameterKDEArea extends SvelteComponent {
  constructor(options) {
    super();
    init$1(this, options, instance$2, create_fragment$3, safe_not_equal, {
      localControlsDivSelector: 0,
      axisMarginLeft: 1,
      axisMarginBottom: 2,
      showAxisX: 3,
      showAxisY: 4,
      data: 11,
      style: 5,
      width: 6,
      height: 7
    });
  }
}
const RedsNoDanger = [
  "primary__energy_red_55",
  "substitute__pink_salmon",
  "primary__energy_red_13",
  "substitute__purple_berry",
  "primary__energy_red_34",
  "substitute__pink_rose",
  "primary__energy_red_21"
];
const Greens = [
  "primary__lichen_green",
  "primary__moss_green_100",
  "primary__moss_green_13",
  "primary__moss_green_21",
  "primary__moss_green_34",
  "primary__moss_green_55",
  "substitute__green_cucumber",
  "substitute__green_mint",
  "substitute__green_succulent"
];
const Blues = [
  "primary__slate_blue",
  "primary__mist_blue",
  "substitute__blue_ocean",
  "substitute__blue_overcast",
  "substitute__blue_sky"
];
const Beiges = ["primary__spruce_wood"];
const _ColorPickingStrategy = [Blues, RedsNoDanger, Greens, Beiges];
function* PickColor() {
  const sourceLists = _ColorPickingStrategy.map((s) => [...s].reverse());
  while (sourceLists.map((s) => s.length).reduce((p, c) => p + c, 0) > 0) {
    for (let list2 of sourceLists) {
      if (list2.length > 0)
        yield list2.pop();
    }
  }
}
const style = getComputedStyle(document.body);
const resolveColor = (key2) => style.getPropertyValue(`--eds_infographic_${key2}`);
const SortedEDSColors = [...PickColor()].map((k) => resolveColor(k));
function create_if_block_1(ctx) {
  let div;
  let t;
  let current;
  let if_block0 = (
    /*option*/
    ctx[28].data.hasHistory && create_if_block_3()
  );
  let if_block1 = (
    /*option*/
    ctx[28].data.hasObservations && create_if_block_2()
  );
  return {
    c() {
      div = element("div");
      if (if_block0)
        if_block0.c();
      t = space();
      if (if_block1)
        if_block1.c();
      attr(div, "class", "flex min-w-8 flex-row w-full justify-evenly");
    },
    m(target, anchor2) {
      insert(target, div, anchor2);
      if (if_block0)
        if_block0.m(div, null);
      append$1(div, t);
      if (if_block1)
        if_block1.m(div, null);
      current = true;
    },
    p(ctx2, dirty) {
      if (
        /*option*/
        ctx2[28].data.hasHistory
      ) {
        if (if_block0) {
          if (dirty & /*option*/
          268435456) {
            transition_in(if_block0, 1);
          }
        } else {
          if_block0 = create_if_block_3();
          if_block0.c();
          transition_in(if_block0, 1);
          if_block0.m(div, t);
        }
      } else if (if_block0) {
        group_outros();
        transition_out(if_block0, 1, 1, () => {
          if_block0 = null;
        });
        check_outros();
      }
      if (
        /*option*/
        ctx2[28].data.hasObservations
      ) {
        if (if_block1) {
          if (dirty & /*option*/
          268435456) {
            transition_in(if_block1, 1);
          }
        } else {
          if_block1 = create_if_block_2();
          if_block1.c();
          transition_in(if_block1, 1);
          if_block1.m(div, null);
        }
      } else if (if_block1) {
        group_outros();
        transition_out(if_block1, 1, 1, () => {
          if_block1 = null;
        });
        check_outros();
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block0);
      transition_in(if_block1);
      current = true;
    },
    o(local) {
      transition_out(if_block0);
      transition_out(if_block1);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div);
      }
      if (if_block0)
        if_block0.d();
      if (if_block1)
        if_block1.d();
    }
  };
}
function create_if_block_3(ctx) {
  let p;
  let icon;
  let t;
  let current;
  icon = new Icon({
    props: {
      width: "22",
      height: "22",
      name: "timeline"
    }
  });
  return {
    c() {
      p = element("p");
      create_component(icon.$$.fragment);
      t = text("   History");
      attr(p, "class", "flex flex-1 justify-end");
    },
    m(target, anchor2) {
      insert(target, p, anchor2);
      mount_component(icon, p, null);
      append$1(p, t);
      current = true;
    },
    i(local) {
      if (current)
        return;
      transition_in(icon.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(icon.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(p);
      }
      destroy_component(icon);
    }
  };
}
function create_if_block_2(ctx) {
  let p;
  let icon;
  let t;
  let current;
  icon = new Icon({
    props: {
      width: "22",
      height: "22",
      name: "table_chart"
    }
  });
  return {
    c() {
      p = element("p");
      create_component(icon.$$.fragment);
      t = text("   Observations");
      attr(p, "class", "flex flex-1 justify-end");
    },
    m(target, anchor2) {
      insert(target, p, anchor2);
      mount_component(icon, p, null);
      append$1(p, t);
      current = true;
    },
    i(local) {
      if (current)
        return;
      transition_in(icon.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(icon.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(p);
      }
      destroy_component(icon);
    }
  };
}
function create_item_slot_1(ctx) {
  let div1;
  let div0;
  let p0;
  let icon;
  let t0;
  let t1_value = (
    /*option*/
    ctx[28].label + ""
  );
  let t1;
  let t2;
  let p1;
  let t3_value = (
    /*option*/
    ctx[28].data.kind + ""
  );
  let t3;
  let t4;
  let current;
  icon = new Icon({
    props: {
      width: "22",
      height: "22",
      name: (
        /*option*/
        ctx[28].data.kind === "summary" ? "time" : "bar_chart"
      )
    }
  });
  let if_block = (
    /*option*/
    (ctx[28].data.hasHistory || /*option*/
    ctx[28].data.hasObservations) && create_if_block_1(ctx)
  );
  return {
    c() {
      div1 = element("div");
      div0 = element("div");
      p0 = element("p");
      create_component(icon.$$.fragment);
      t0 = text(" ");
      t1 = text(t1_value);
      t2 = space();
      p1 = element("p");
      t3 = text(t3_value);
      t4 = space();
      if (if_block)
        if_block.c();
      attr(p0, "class", "flex flex-1");
      attr(p1, "class", "font-light ml-2 italic");
      attr(div0, "class", "flex flex-col w-full");
      attr(div1, "slot", "item");
      attr(div1, "class", "w-full p-3 flex flex-row");
    },
    m(target, anchor2) {
      insert(target, div1, anchor2);
      append$1(div1, div0);
      append$1(div0, p0);
      mount_component(icon, p0, null);
      append$1(p0, t0);
      append$1(p0, t1);
      append$1(div0, t2);
      append$1(div0, p1);
      append$1(p1, t3);
      append$1(div1, t4);
      if (if_block)
        if_block.m(div1, null);
      current = true;
    },
    p(ctx2, dirty) {
      const icon_changes = {};
      if (dirty & /*option*/
      268435456)
        icon_changes.name = /*option*/
        ctx2[28].data.kind === "summary" ? "time" : "bar_chart";
      icon.$set(icon_changes);
      if ((!current || dirty & /*option*/
      268435456) && t1_value !== (t1_value = /*option*/
      ctx2[28].label + ""))
        set_data(t1, t1_value);
      if ((!current || dirty & /*option*/
      268435456) && t3_value !== (t3_value = /*option*/
      ctx2[28].data.kind + ""))
        set_data(t3, t3_value);
      if (
        /*option*/
        ctx2[28].data.hasHistory || /*option*/
        ctx2[28].data.hasObservations
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
          if (dirty & /*option*/
          268435456) {
            transition_in(if_block, 1);
          }
        } else {
          if_block = create_if_block_1(ctx2);
          if_block.c();
          transition_in(if_block, 1);
          if_block.m(div1, null);
        }
      } else if (if_block) {
        group_outros();
        transition_out(if_block, 1, 1, () => {
          if_block = null;
        });
        check_outros();
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(icon.$$.fragment, local);
      transition_in(if_block);
      current = true;
    },
    o(local) {
      transition_out(icon.$$.fragment, local);
      transition_out(if_block);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div1);
      }
      destroy_component(icon);
      if (if_block)
        if_block.d();
    }
  };
}
function create_item_slot(ctx) {
  let div;
  let t_value = (
    /*option*/
    ctx[28].label + ""
  );
  let t;
  return {
    c() {
      div = element("div");
      t = text(t_value);
      attr(div, "slot", "item");
      attr(div, "class", "w-full p-3 flex flex-row");
    },
    m(target, anchor2) {
      insert(target, div, anchor2);
      append$1(div, t);
    },
    p(ctx2, dirty) {
      if (dirty & /*option*/
      268435456 && t_value !== (t_value = /*option*/
      ctx2[28].label + ""))
        set_data(t, t_value);
    },
    d(detaching) {
      if (detaching) {
        detach(div);
      }
    }
  };
}
function create_controls_slot(ctx) {
  let div1;
  let combobox0;
  let t0;
  let br0;
  let t1;
  let combobox1;
  let t2;
  let br1;
  let t3;
  let combobox2;
  let t4;
  let div0;
  let div0_id_value;
  let current;
  combobox0 = new Combobox({
    props: {
      label: "Ensembles",
      placeholder: "Select ensembles…",
      options: (
        /*experiment*/
        ctx[7]().sortedEnsembles().map(func)
      ),
      customValueLabel: (
        /*ensembleLabelFunction*/
        ctx[16]
      ),
      value: (
        /*spec*/
        ctx[2].query.ensembles.map(
          /*func_1*/
          ctx[21]
        )
      ),
      multiselect: true
    }
  });
  combobox0.$on(
    "change",
    /*change_handler*/
    ctx[22]
  );
  combobox1 = new Combobox({
    props: {
      label: "Keyword",
      placeholder: "Select keyword…",
      options: (
        /*experiment*/
        ctx[7]().availableKeywords().map(func_2)
      ),
      value: (
        /*spec*/
        ctx[2].query.keyword
      ),
      noMinItemHeight: true,
      $$slots: {
        item: [
          create_item_slot_1,
          ({ option }) => ({ 28: option }),
          ({ option }) => option ? 268435456 : 0
        ]
      },
      $$scope: { ctx }
    }
  });
  combobox1.$on(
    "change",
    /*change_handler_1*/
    ctx[23]
  );
  combobox2 = new Combobox({
    props: {
      label: "Chart",
      placeholder: "Select chart type",
      options: (
        /*availableChartTypes*/
        ctx[6].map(func_3)
      ),
      value: (
        /*spec*/
        ctx[2].chart
      ),
      noMinItemHeight: true,
      $$slots: {
        item: [
          create_item_slot,
          ({ option }) => ({ 28: option }),
          ({ option }) => option ? 268435456 : 0
        ]
      },
      $$scope: { ctx }
    }
  });
  combobox2.$on(
    "change",
    /*change_handler_2*/
    ctx[24]
  );
  return {
    c() {
      div1 = element("div");
      create_component(combobox0.$$.fragment);
      t0 = space();
      br0 = element("br");
      t1 = space();
      create_component(combobox1.$$.fragment);
      t2 = space();
      br1 = element("br");
      t3 = space();
      create_component(combobox2.$$.fragment);
      t4 = space();
      div0 = element("div");
      attr(div0, "class", "p-3");
      attr(div0, "id", div0_id_value = /*localControlsDivId*/
      ctx[8]());
      attr(div1, "slot", "controls");
      attr(div1, "class", "p-2");
    },
    m(target, anchor2) {
      insert(target, div1, anchor2);
      mount_component(combobox0, div1, null);
      append$1(div1, t0);
      append$1(div1, br0);
      append$1(div1, t1);
      mount_component(combobox1, div1, null);
      append$1(div1, t2);
      append$1(div1, br1);
      append$1(div1, t3);
      mount_component(combobox2, div1, null);
      append$1(div1, t4);
      append$1(div1, div0);
      current = true;
    },
    p(ctx2, dirty) {
      const combobox0_changes = {};
      if (dirty & /*experiment*/
      128)
        combobox0_changes.options = /*experiment*/
        ctx2[7]().sortedEnsembles().map(func);
      if (dirty & /*spec, experiment*/
      132)
        combobox0_changes.value = /*spec*/
        ctx2[2].query.ensembles.map(
          /*func_1*/
          ctx2[21]
        );
      combobox0.$set(combobox0_changes);
      const combobox1_changes = {};
      if (dirty & /*experiment*/
      128)
        combobox1_changes.options = /*experiment*/
        ctx2[7]().availableKeywords().map(func_2);
      if (dirty & /*spec*/
      4)
        combobox1_changes.value = /*spec*/
        ctx2[2].query.keyword;
      if (dirty & /*$$scope, option*/
      805306368) {
        combobox1_changes.$$scope = { dirty, ctx: ctx2 };
      }
      combobox1.$set(combobox1_changes);
      const combobox2_changes = {};
      if (dirty & /*availableChartTypes*/
      64)
        combobox2_changes.options = /*availableChartTypes*/
        ctx2[6].map(func_3);
      if (dirty & /*spec*/
      4)
        combobox2_changes.value = /*spec*/
        ctx2[2].chart;
      if (dirty & /*$$scope, option*/
      805306368) {
        combobox2_changes.$$scope = { dirty, ctx: ctx2 };
      }
      combobox2.$set(combobox2_changes);
      if (!current || dirty & /*localControlsDivId*/
      256 && div0_id_value !== (div0_id_value = /*localControlsDivId*/
      ctx2[8]())) {
        attr(div0, "id", div0_id_value);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(combobox0.$$.fragment, local);
      transition_in(combobox1.$$.fragment, local);
      transition_in(combobox2.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(combobox0.$$.fragment, local);
      transition_out(combobox1.$$.fragment, local);
      transition_out(combobox2.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div1);
      }
      destroy_component(combobox0);
      destroy_component(combobox1);
      destroy_component(combobox2);
    }
  };
}
function create_catch_block_1(ctx) {
  return { c: noop$2, m: noop$2, p: noop$2, d: noop$2 };
}
function create_then_block_1(ctx) {
  let div2;
  let div0;
  let t0_value = (
    /*spec*/
    ctx[2].chart + ""
  );
  let t0;
  let t1;
  let t2_value = (
    /*spec*/
    ctx[2].kind + ""
  );
  let t2;
  let t3;
  let t4_value = (
    /*spec*/
    ctx[2].query.keyword + ""
  );
  let t4;
  let t5;
  let div1;
  let t6_value = (
    /*performanceInfo*/
    ctx[9]() + ""
  );
  let t6;
  return {
    c() {
      div2 = element("div");
      div0 = element("div");
      t0 = text(t0_value);
      t1 = text(" of ");
      t2 = text(t2_value);
      t3 = text(" keyword ");
      t4 = text(t4_value);
      t5 = space();
      div1 = element("div");
      t6 = text(t6_value);
      attr(div0, "class", "flex flex-1 justify-end text-lg");
      attr(div1, "class", "flex flex-1 justify-end italic font-thin");
      attr(div2, "class", "flex-1 justify-end pr-8");
    },
    m(target, anchor2) {
      insert(target, div2, anchor2);
      append$1(div2, div0);
      append$1(div0, t0);
      append$1(div0, t1);
      append$1(div0, t2);
      append$1(div0, t3);
      append$1(div0, t4);
      append$1(div2, t5);
      append$1(div2, div1);
      append$1(div1, t6);
    },
    p(ctx2, dirty) {
      if (dirty & /*spec*/
      4 && t0_value !== (t0_value = /*spec*/
      ctx2[2].chart + ""))
        set_data(t0, t0_value);
      if (dirty & /*spec*/
      4 && t2_value !== (t2_value = /*spec*/
      ctx2[2].kind + ""))
        set_data(t2, t2_value);
      if (dirty & /*spec*/
      4 && t4_value !== (t4_value = /*spec*/
      ctx2[2].query.keyword + ""))
        set_data(t4, t4_value);
      if (dirty & /*performanceInfo*/
      512 && t6_value !== (t6_value = /*performanceInfo*/
      ctx2[9]() + ""))
        set_data(t6, t6_value);
    },
    d(detaching) {
      if (detaching) {
        detach(div2);
      }
    }
  };
}
function create_pending_block_1(ctx) {
  let p;
  return {
    c() {
      p = element("p");
      p.textContent = "Loading data...";
    },
    m(target, anchor2) {
      insert(target, p, anchor2);
    },
    p: noop$2,
    d(detaching) {
      if (detaching) {
        detach(p);
      }
    }
  };
}
function create_catch_block$1(ctx) {
  return {
    c: noop$2,
    m: noop$2,
    p: noop$2,
    i: noop$2,
    o: noop$2,
    d: noop$2
  };
}
function create_then_block$1(ctx) {
  let if_block_anchor;
  let current;
  let if_block = (
    /*containerWidth*/
    ctx[3] > 0 && /*containerHeight*/
    ctx[4] > 0 && create_if_block(ctx)
  );
  return {
    c() {
      if (if_block)
        if_block.c();
      if_block_anchor = empty$1();
    },
    m(target, anchor2) {
      if (if_block)
        if_block.m(target, anchor2);
      insert(target, if_block_anchor, anchor2);
      current = true;
    },
    p(ctx2, dirty) {
      if (
        /*containerWidth*/
        ctx2[3] > 0 && /*containerHeight*/
        ctx2[4] > 0
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
          if (dirty & /*containerWidth, containerHeight*/
          24) {
            transition_in(if_block, 1);
          }
        } else {
          if_block = create_if_block(ctx2);
          if_block.c();
          transition_in(if_block, 1);
          if_block.m(if_block_anchor.parentNode, if_block_anchor);
        }
      } else if (if_block) {
        group_outros();
        transition_out(if_block, 1, 1, () => {
          if_block = null;
        });
        check_outros();
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block);
      current = true;
    },
    o(local) {
      transition_out(if_block);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(if_block_anchor);
      }
      if (if_block)
        if_block.d(detaching);
    }
  };
}
function create_if_block(ctx) {
  let switch_instance;
  let switch_instance_anchor;
  let current;
  var switch_value = (
    /*findComponentAndDataInfo*/
    ctx[13]().component
  );
  function switch_props(ctx2, dirty) {
    return {
      props: {
        data: (
          /*findComponentAndDataInfo*/
          ctx2[13]().data
        ),
        style: (
          /*computeEnsembleStyleMap*/
          ctx2[10]()
        ),
        width: (
          /*containerWidth*/
          ctx2[3] - 20
        ),
        height: (
          /*containerHeight*/
          ctx2[4] - 20
        ),
        axisMarginLeft: (
          /*axisMarginLeft*/
          ctx2[0]
        ),
        axisMarginBottom: (
          /*axisMarginBottom*/
          ctx2[1]
        ),
        localControlsDivSelector: "#" + /*localControlsDivId*/
        ctx2[8]()
      }
    };
  }
  if (switch_value) {
    switch_instance = construct_svelte_component(switch_value, switch_props(ctx));
  }
  return {
    c() {
      if (switch_instance)
        create_component(switch_instance.$$.fragment);
      switch_instance_anchor = empty$1();
    },
    m(target, anchor2) {
      if (switch_instance)
        mount_component(switch_instance, target, anchor2);
      insert(target, switch_instance_anchor, anchor2);
      current = true;
    },
    p(ctx2, dirty) {
      if (switch_value !== (switch_value = /*findComponentAndDataInfo*/
      ctx2[13]().component)) {
        if (switch_instance) {
          group_outros();
          const old_component = switch_instance;
          transition_out(old_component.$$.fragment, 1, 0, () => {
            destroy_component(old_component, 1);
          });
          check_outros();
        }
        if (switch_value) {
          switch_instance = construct_svelte_component(switch_value, switch_props(ctx2));
          create_component(switch_instance.$$.fragment);
          transition_in(switch_instance.$$.fragment, 1);
          mount_component(switch_instance, switch_instance_anchor.parentNode, switch_instance_anchor);
        } else {
          switch_instance = null;
        }
      } else if (switch_value) {
        const switch_instance_changes = {};
        if (dirty & /*computeEnsembleStyleMap*/
        1024)
          switch_instance_changes.style = /*computeEnsembleStyleMap*/
          ctx2[10]();
        if (dirty & /*containerWidth*/
        8)
          switch_instance_changes.width = /*containerWidth*/
          ctx2[3] - 20;
        if (dirty & /*containerHeight*/
        16)
          switch_instance_changes.height = /*containerHeight*/
          ctx2[4] - 20;
        if (dirty & /*axisMarginLeft*/
        1)
          switch_instance_changes.axisMarginLeft = /*axisMarginLeft*/
          ctx2[0];
        if (dirty & /*axisMarginBottom*/
        2)
          switch_instance_changes.axisMarginBottom = /*axisMarginBottom*/
          ctx2[1];
        if (dirty & /*localControlsDivId*/
        256)
          switch_instance_changes.localControlsDivSelector = "#" + /*localControlsDivId*/
          ctx2[8]();
        switch_instance.$set(switch_instance_changes);
      }
    },
    i(local) {
      if (current)
        return;
      if (switch_instance)
        transition_in(switch_instance.$$.fragment, local);
      current = true;
    },
    o(local) {
      if (switch_instance)
        transition_out(switch_instance.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(switch_instance_anchor);
      }
      if (switch_instance)
        destroy_component(switch_instance, detaching);
    }
  };
}
function create_pending_block$1(ctx) {
  let p;
  return {
    c() {
      p = element("p");
      p.textContent = "Loading datasets...";
    },
    m(target, anchor2) {
      insert(target, p, anchor2);
    },
    p: noop$2,
    i: noop$2,
    o: noop$2,
    d(detaching) {
      if (detaching) {
        detach(p);
      }
    }
  };
}
function create_fragment$2(ctx) {
  let div1;
  let span;
  let sidebar;
  let updating_open;
  let updating_experiment;
  let t0;
  let promise;
  let t1;
  let div0;
  let promise_1;
  let current;
  let mounted;
  let dispose;
  function sidebar_open_binding(value) {
    ctx[25](value);
  }
  function sidebar_experiment_binding(value) {
    ctx[26](value);
  }
  let sidebar_props = {
    $$slots: { controls: [create_controls_slot] },
    $$scope: { ctx }
  };
  if (
    /*isSidebarOpen*/
    ctx[5] !== void 0
  ) {
    sidebar_props.open = /*isSidebarOpen*/
    ctx[5];
  }
  if (
    /*spec*/
    ctx[2].query.experiment !== void 0
  ) {
    sidebar_props.experiment = /*spec*/
    ctx[2].query.experiment;
  }
  sidebar = new Sidebar({ props: sidebar_props });
  binding_callbacks.push(() => bind(sidebar, "open", sidebar_open_binding));
  binding_callbacks.push(() => bind(sidebar, "experiment", sidebar_experiment_binding));
  let info = {
    ctx,
    current: null,
    token: null,
    hasCatch: false,
    pending: create_pending_block_1,
    then: create_then_block_1,
    catch: create_catch_block_1
  };
  handle_promise(
    promise = Promise.all([
      /*ensureComponentDataIsLoaded*/
      ctx[11](),
      /*ensureExperimentIsSelected*/
      ctx[12]()
    ]),
    info
  );
  let info_1 = {
    ctx,
    current: null,
    token: null,
    hasCatch: false,
    pending: create_pending_block$1,
    then: create_then_block$1,
    catch: create_catch_block$1,
    blocks: [, , ,]
  };
  handle_promise(
    promise_1 = Promise.all([
      /*ensureComponentDataIsLoaded*/
      ctx[11](),
      /*ensureExperimentIsSelected*/
      ctx[12]()
    ]),
    info_1
  );
  return {
    c() {
      div1 = element("div");
      span = element("span");
      create_component(sidebar.$$.fragment);
      t0 = space();
      info.block.c();
      t1 = space();
      div0 = element("div");
      info_1.block.c();
      attr(span, "class", "inline-flex p-3 w-full");
      attr(div0, "class", "h-96 relative");
      attr(div1, "class", "relative w-full h-full shadow-lg rounded m-5");
    },
    m(target, anchor2) {
      insert(target, div1, anchor2);
      append$1(div1, span);
      mount_component(sidebar, span, null);
      append$1(span, t0);
      info.block.m(span, info.anchor = null);
      info.mount = () => span;
      info.anchor = null;
      append$1(div1, t1);
      append$1(div1, div0);
      info_1.block.m(div0, info_1.anchor = null);
      info_1.mount = () => div0;
      info_1.anchor = null;
      current = true;
      if (!mounted) {
        dispose = [
          action_destroyer(watchResize.call(null, div0)),
          listen(
            div0,
            "resized",
            /*onResize*/
            ctx[14]
          )
        ];
        mounted = true;
      }
    },
    p(new_ctx, [dirty]) {
      ctx = new_ctx;
      const sidebar_changes = {};
      if (dirty & /*$$scope, localControlsDivId, availableChartTypes, spec, experiment*/
      536871364) {
        sidebar_changes.$$scope = { dirty, ctx };
      }
      if (!updating_open && dirty & /*isSidebarOpen*/
      32) {
        updating_open = true;
        sidebar_changes.open = /*isSidebarOpen*/
        ctx[5];
        add_flush_callback(() => updating_open = false);
      }
      if (!updating_experiment && dirty & /*spec*/
      4) {
        updating_experiment = true;
        sidebar_changes.experiment = /*spec*/
        ctx[2].query.experiment;
        add_flush_callback(() => updating_experiment = false);
      }
      sidebar.$set(sidebar_changes);
      info.ctx = ctx;
      if (dirty & /*ensureComponentDataIsLoaded*/
      2048 && promise !== (promise = Promise.all([
        /*ensureComponentDataIsLoaded*/
        ctx[11](),
        /*ensureExperimentIsSelected*/
        ctx[12]()
      ])) && handle_promise(promise, info))
        ;
      else {
        update_await_block_branch(info, ctx, dirty);
      }
      info_1.ctx = ctx;
      if (dirty & /*ensureComponentDataIsLoaded*/
      2048 && promise_1 !== (promise_1 = Promise.all([
        /*ensureComponentDataIsLoaded*/
        ctx[11](),
        /*ensureExperimentIsSelected*/
        ctx[12]()
      ])) && handle_promise(promise_1, info_1))
        ;
      else {
        update_await_block_branch(info_1, ctx, dirty);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(sidebar.$$.fragment, local);
      transition_in(info_1.block);
      current = true;
    },
    o(local) {
      transition_out(sidebar.$$.fragment, local);
      for (let i2 = 0; i2 < 3; i2 += 1) {
        const block2 = info_1.blocks[i2];
        transition_out(block2);
      }
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div1);
      }
      destroy_component(sidebar);
      info.block.d();
      info.token = null;
      info = null;
      info_1.block.d();
      info_1.token = null;
      info_1 = null;
      mounted = false;
      run_all(dispose);
    }
  };
}
const func = (ens) => ({
  value: ens.id,
  label: `Iteration ${ens.iteration}`
});
const func_2 = (info) => ({
  value: info.key,
  label: info.key,
  data: info
});
const func_3 = (t) => ({ value: t, label: t });
function instance$1($$self, $$props, $$invalidate) {
  let spec;
  let ensureComponentDataIsLoaded;
  let computeEnsembleStyleMap;
  let performanceInfo;
  let experiment;
  let localControlsDivId;
  let availableChartTypes;
  let $plotterStore;
  component_subscribe($$self, plotterStore, ($$value) => $$invalidate(20, $plotterStore = $$value));
  let { viewIndex } = $$props;
  let { axisMarginLeft = 65 } = $$props;
  let { axisMarginBottom = 35 } = $$props;
  let { showAxisX = true } = $$props;
  let { showAxisY = true } = $$props;
  const ensureExperimentIsSelected = async () => {
    await fetchExperiments();
    const metadata = getLoadedExperiments();
    if (spec.query.experiment === "auto")
      $$invalidate(2, spec.query.experiment = Object.keys(metadata)[0], spec);
  };
  const DataToComponentType = [
    {
      kind: "parameter",
      chart: "ridgelines",
      component: ParameterKDERidgelines
    },
    {
      kind: "parameter",
      chart: "area",
      component: ParameterKDEArea
    },
    {
      kind: "summary",
      chart: "line",
      component: SummaryLines
    }
  ];
  const findComponentAndDataInfo = () => {
    const { kind, chart } = spec;
    const component = DataToComponentType.find((v) => v.kind === kind && v.chart === chart).component;
    if (kind === "parameter") {
      const param = getLoadedParameter(spec.query);
      return { component, ...param };
    }
    if (kind === "summary") {
      const summary = getLoadedSummary(spec.query);
      return { component, ...summary };
    }
    throw new TypeError(`Expected valid dataset kind, got ${kind}, expected one of [${["summary", "parameter"].join(", ")}]`);
  };
  let containerWidth = 0;
  let containerHeight = 0;
  const onResize = (event) => {
    const { entries } = event.detail;
    const entry = entries[0];
    const { width, height } = entry.contentRect;
    $$invalidate(3, containerWidth = width);
    $$invalidate(4, containerHeight = height);
  };
  let isSidebarOpen = false;
  const updateChartQuery = async (keyword) => {
    const info = experiment().getKeywordInfo(keyword);
    const { kind } = info;
    if (kind !== spec.kind) {
      const chart = DataToComponentType.find((info2) => info2.kind === kind).chart;
      Object.assign(spec, { chart, kind });
      $$invalidate(2, spec.query.keyword = keyword, spec);
    } else {
      $$invalidate(2, spec.query.keyword = keyword, spec);
    }
  };
  const ensembleLabelFunction = (value, options) => {
    const singleLabels = value.map((v) => options.find((o) => o.value === v).label);
    const niceLabel = `Iterations ${singleLabels.map((sl) => sl.replace("Iteration ", "")).sort().join(", ")}`;
    return niceLabel;
  };
  const func_12 = (e) => experiment().ensembleAliasToId(e);
  const change_handler = (e) => $$invalidate(2, spec.query.ensembles = e.detail.value, spec);
  const change_handler_1 = (e) => updateChartQuery(e.detail.value);
  const change_handler_2 = (e) => $$invalidate(2, spec.chart = e.detail.value, spec);
  function sidebar_open_binding(value) {
    isSidebarOpen = value;
    $$invalidate(5, isSidebarOpen);
  }
  function sidebar_experiment_binding(value) {
    if ($$self.$$.not_equal(spec.query.experiment, value)) {
      spec.query.experiment = value;
      $$invalidate(2, spec), $$invalidate(20, $plotterStore), $$invalidate(17, viewIndex);
    }
  }
  $$self.$$set = ($$props2) => {
    if ("viewIndex" in $$props2)
      $$invalidate(17, viewIndex = $$props2.viewIndex);
    if ("axisMarginLeft" in $$props2)
      $$invalidate(0, axisMarginLeft = $$props2.axisMarginLeft);
    if ("axisMarginBottom" in $$props2)
      $$invalidate(1, axisMarginBottom = $$props2.axisMarginBottom);
    if ("showAxisX" in $$props2)
      $$invalidate(18, showAxisX = $$props2.showAxisX);
    if ("showAxisY" in $$props2)
      $$invalidate(19, showAxisY = $$props2.showAxisY);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty & /*$plotterStore, viewIndex*/
    1179648) {
      $$invalidate(2, spec = $plotterStore.charts[viewIndex]);
    }
    if ($$self.$$.dirty & /*spec*/
    4) {
      $$invalidate(11, ensureComponentDataIsLoaded = async () => {
        const { kind, query } = spec;
        if (query.ensembles.length === 0)
          return true;
        switch (kind) {
          case "summary":
            await fetchSummary(query);
            break;
          case "parameter":
            await fetchParameter(query);
            break;
        }
      });
    }
    if ($$self.$$.dirty & /*spec, $plotterStore*/
    1048580) {
      $$invalidate(10, computeEnsembleStyleMap = () => {
        const style2 = spec.style || {};
        const stylePerEnsemble = {};
        const experiment2 = getLoadedExperiments()[spec.query.experiment];
        experiment2.eachEnsemble((ensembleId, ensembleAlias, index) => {
          stylePerEnsemble[ensembleAlias] = stylePerEnsemble[ensembleId] = {
            stroke: SortedEDSColors[index],
            fill: SortedEDSColors[index]
          };
          const globalStyle = $plotterStore.style;
          Object.keys(globalStyle).filter((k) => k.startsWith("ensemble:") || k === "*").filter((k) => k === ensembleAlias || k === "*").forEach((k) => Object.assign(stylePerEnsemble[ensembleAlias], globalStyle[k]));
          Object.keys(style2 || {}).filter((k) => k.startsWith("ensemble:") || k === "*").filter((k) => k === ensembleAlias || k === "*").forEach((k) => Object.assign(stylePerEnsemble[ensembleAlias], style2[k]));
        });
        return stylePerEnsemble;
      });
    }
    if ($$self.$$.dirty & /*spec*/
    4) {
      $$invalidate(7, experiment = () => getLoadedExperiments()[spec.query.experiment]);
    }
    if ($$self.$$.dirty & /*viewIndex*/
    131072) {
      $$invalidate(8, localControlsDivId = () => `chart-view-${viewIndex}-local-controls`);
    }
    if ($$self.$$.dirty & /*spec*/
    4) {
      $$invalidate(6, availableChartTypes = DataToComponentType.filter((info) => info.kind === spec.kind).map((info) => info.chart));
    }
  };
  $$invalidate(9, performanceInfo = () => {
    try {
      const summary = findComponentAndDataInfo();
      const { timeSpentSeconds, MBProcessed } = summary;
      return `Loaded ${summary.data.length} realizations in ${timeSpentSeconds.toFixed(4)}s, ${MBProcessed}MB processed`;
    } catch (e) {
      return `Loading data...`;
    }
  });
  return [
    axisMarginLeft,
    axisMarginBottom,
    spec,
    containerWidth,
    containerHeight,
    isSidebarOpen,
    availableChartTypes,
    experiment,
    localControlsDivId,
    performanceInfo,
    computeEnsembleStyleMap,
    ensureComponentDataIsLoaded,
    ensureExperimentIsSelected,
    findComponentAndDataInfo,
    onResize,
    updateChartQuery,
    ensembleLabelFunction,
    viewIndex,
    showAxisX,
    showAxisY,
    $plotterStore,
    func_12,
    change_handler,
    change_handler_1,
    change_handler_2,
    sidebar_open_binding,
    sidebar_experiment_binding
  ];
}
class GeneralChartView extends SvelteComponent {
  constructor(options) {
    super();
    init$1(this, options, instance$1, create_fragment$2, safe_not_equal, {
      viewIndex: 17,
      axisMarginLeft: 0,
      axisMarginBottom: 1,
      showAxisX: 18,
      showAxisY: 19
    });
  }
}
function get_each_context(ctx, list2, i2) {
  const child_ctx = ctx.slice();
  child_ctx[1] = list2[i2];
  child_ctx[3] = i2;
  return child_ctx;
}
function create_catch_block(ctx) {
  return {
    c: noop$2,
    m: noop$2,
    p: noop$2,
    i: noop$2,
    o: noop$2,
    d: noop$2
  };
}
function create_then_block(ctx) {
  let each_1_anchor;
  let current;
  let each_value = ensure_array_like({
    length: (
      /*$plotterStore*/
      ctx[0].charts.length
    )
  });
  let each_blocks = [];
  for (let i2 = 0; i2 < each_value.length; i2 += 1) {
    each_blocks[i2] = create_each_block(get_each_context(ctx, each_value, i2));
  }
  const out = (i2) => transition_out(each_blocks[i2], 1, 1, () => {
    each_blocks[i2] = null;
  });
  return {
    c() {
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        each_blocks[i2].c();
      }
      each_1_anchor = empty$1();
    },
    m(target, anchor2) {
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        if (each_blocks[i2]) {
          each_blocks[i2].m(target, anchor2);
        }
      }
      insert(target, each_1_anchor, anchor2);
      current = true;
    },
    p(ctx2, dirty) {
      if (dirty & /*$plotterStore*/
      1) {
        each_value = ensure_array_like({
          length: (
            /*$plotterStore*/
            ctx2[0].charts.length
          )
        });
        let i2;
        for (i2 = 0; i2 < each_value.length; i2 += 1) {
          const child_ctx = get_each_context(ctx2, each_value, i2);
          if (each_blocks[i2]) {
            each_blocks[i2].p(child_ctx, dirty);
            transition_in(each_blocks[i2], 1);
          } else {
            each_blocks[i2] = create_each_block(child_ctx);
            each_blocks[i2].c();
            transition_in(each_blocks[i2], 1);
            each_blocks[i2].m(each_1_anchor.parentNode, each_1_anchor);
          }
        }
        group_outros();
        for (i2 = each_value.length; i2 < each_blocks.length; i2 += 1) {
          out(i2);
        }
        check_outros();
      }
    },
    i(local) {
      if (current)
        return;
      for (let i2 = 0; i2 < each_value.length; i2 += 1) {
        transition_in(each_blocks[i2]);
      }
      current = true;
    },
    o(local) {
      each_blocks = each_blocks.filter(Boolean);
      for (let i2 = 0; i2 < each_blocks.length; i2 += 1) {
        transition_out(each_blocks[i2]);
      }
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(each_1_anchor);
      }
      destroy_each(each_blocks, detaching);
    }
  };
}
function create_each_block(ctx) {
  let generalchartview;
  let current;
  generalchartview = new GeneralChartView({ props: { viewIndex: (
    /*index*/
    ctx[3]
  ) } });
  return {
    c() {
      create_component(generalchartview.$$.fragment);
    },
    m(target, anchor2) {
      mount_component(generalchartview, target, anchor2);
      current = true;
    },
    p: noop$2,
    i(local) {
      if (current)
        return;
      transition_in(generalchartview.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(generalchartview.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(generalchartview, detaching);
    }
  };
}
function create_pending_block(ctx) {
  let p;
  return {
    c() {
      p = element("p");
      p.textContent = "Fetching experiments...";
    },
    m(target, anchor2) {
      insert(target, p, anchor2);
    },
    p: noop$2,
    i: noop$2,
    o: noop$2,
    d(detaching) {
      if (detaching) {
        detach(p);
      }
    }
  };
}
function create_fragment$1(ctx) {
  let div;
  let current;
  let info = {
    ctx,
    current: null,
    token: null,
    hasCatch: false,
    pending: create_pending_block,
    then: create_then_block,
    catch: create_catch_block,
    blocks: [, , ,]
  };
  handle_promise(ensureStoreIsSyncedWithExperiments(), info);
  return {
    c() {
      div = element("div");
      info.block.c();
      attr(div, "class", "flex-wrap flex m-2");
    },
    m(target, anchor2) {
      insert(target, div, anchor2);
      info.block.m(div, info.anchor = null);
      info.mount = () => div;
      info.anchor = null;
      current = true;
    },
    p(new_ctx, [dirty]) {
      ctx = new_ctx;
      update_await_block_branch(info, ctx, dirty);
    },
    i(local) {
      if (current)
        return;
      transition_in(info.block);
      current = true;
    },
    o(local) {
      for (let i2 = 0; i2 < 3; i2 += 1) {
        const block2 = info.blocks[i2];
        transition_out(block2);
      }
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div);
      }
      info.block.d();
      info.token = null;
      info = null;
    }
  };
}
function instance($$self, $$props, $$invalidate) {
  let $plotterStore;
  component_subscribe($$self, plotterStore, ($$value) => $$invalidate(0, $plotterStore = $$value));
  return [$plotterStore];
}
class MainView extends SvelteComponent {
  constructor(options) {
    super();
    init$1(this, options, instance, create_fragment$1, safe_not_equal, {});
  }
}
function create_fragment(ctx) {
  let main;
  let mainview;
  let current;
  mainview = new MainView({});
  return {
    c() {
      main = element("main");
      create_component(mainview.$$.fragment);
    },
    m(target, anchor2) {
      insert(target, main, anchor2);
      mount_component(mainview, main, null);
      current = true;
    },
    p: noop$2,
    i(local) {
      if (current)
        return;
      transition_in(mainview.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(mainview.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(main);
      }
      destroy_component(mainview);
    }
  };
}
class App extends SvelteComponent {
  constructor(options) {
    super();
    init$1(this, options, null, create_fragment, safe_not_equal, {});
  }
}
new App({
  target: document.getElementById("app")
});
//# sourceMappingURL=index-PhcCE_Qm.js.map
