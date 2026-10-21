// Local frontend QA. Requires Playwright and an existing Chromium browser; no downloads.
const { chromium } = require('playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const { createHash } = require('node:crypto');
const root = path.resolve(__dirname, '..');
const url = process.env.COMFY_URL || 'http://127.0.0.1:8291';
const variants = process.argv.includes('--base') ? ['base'] : process.argv.includes('--sft') ? ['sft'] : ['base', 'sft'];
const resumeIndex = process.argv.indexOf('--resume');
const resumeId = resumeIndex < 0 ? null : process.argv[resumeIndex + 1];

(async () => {
  if ((process.argv.includes('--run') || resumeId) && variants.length !== 1) {
    throw new Error('Select --base or --sft for a canvas execution.');
  }
  const sourceHashes = async () => Object.fromEntries(await Promise.all(
    ['cache.py', 'runtime.py', 'nodes.py'].map(async name =>
      [name, createHash('sha256').update(await fs.readFile(path.join(root, name))).digest('hex')])
  ));
  const source = await sourceHashes();
  const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_PATH || undefined});
  try {
    const page = await browser.newPage({viewport: {width: 1800, height: 1200}});
    const errors = [];
    page.on('pageerror', e => errors.push(String(e)));
    await page.goto(url, {waitUntil: 'networkidle'});
    await page.waitForFunction(() => !!window.app?.graph && !!window.LiteGraph, {timeout: 60000});
    await fs.mkdir(path.join(root, 'workflows'), {recursive: true});
    await fs.mkdir(path.join(root, 'artifacts'), {recursive: true});
    for (const variant of variants) {
      const workflow = await page.evaluate((variant) => {
        const app = window.app;
        app.graph.clear();
        function node(type, title, pos, values = {}, size) {
          const n = window.LiteGraph.createNode(type);
          if (!n) throw new Error('Missing frontend node: ' + type);
          app.graph.add(n);
          n.title = title;
          n.pos = pos;
          for (const [name, value] of Object.entries(values)) {
            const widget = n.widgets?.find(w => w.name === name);
            if (!widget) throw new Error(type + ': missing widget ' + name);
            widget.value = value;
          }
          n.properties['Node name for S&R'] = type;
          if (size) n.size = size;
          return n;
        }
        const dit = node('UNETLoader', '01 / Native MOSS ' + variant.toUpperCase(), [30, 60], {
          unet_name: `moss_image_original_${variant}_bf16.safetensors`, weight_dtype: 'default'
        }, [360, 100]);
        const clip = node('CLIPLoader', '02 / Qwen3.5 native text encoder', [30, 230], {
          clip_name: 'moss_qwen35_9b_bf16_native.safetensors', type: 'moss_image', device: 'default'
        }, [360, 130]);
        const vae = node('VAELoader', '03 / MOSS VAE', [30, 440], {
          vae_name: 'moss_image_official_vae.safetensors'
        }, [360, 80]);
        const latent = node('EmptyFlux2LatentImage', '04 / Native 128-channel latent', [30, 610], {
          width: 1024, height: 1024, batch_size: 1
        }, [360, 130]);
        const positive = node('CLIPTextEncode', '05 / Positive prompt', [440, 60], {
          text: 'A clean studio photograph of a white enamel sign with large centered dark blue text exactly reading "MOSS IMAGE" on the first line and "OPEN 09:00" on the second line. A small green plant beside the sign, neutral wall, crisp typography.'
        }, [380, 240]);
        const negative = node('CLIPTextEncode', '06 / Empty negative prompt', [440, 370], {text: ''}, [380, 150]);
        const cache = node('MOSSImageBlockCacheT8', '07 / T8 BlockCache / compare ON and OFF', [440, 600], {
          enabled: true, threshold: 0.08, start_percent: 0.10, end_percent: 0.90, max_consecutive_hits: 1,
          cache_device: 'gpu', max_cache_mb: 512, metric_stride: 4,
          threshold_mode: 'constant', split_ratio: 0.30, late_threshold: 0.14
        }, [380, 360]);
        const slg = node('SkipLayerGuidanceDiTSimple', '08 / Negative-branch skip22', [890, 640], {
          double_layers: '', single_layers: '22', start_percent: 0, end_percent: 1
        }, [340, 190]);
        const sampler = node('KSampler', '09 / Native sampling / fixed seed', [890, 60], {
          seed: 42, control_after_generate: 'fixed', steps: 50, cfg: 3, sampler_name: 'euler', scheduler: 'simple', denoise: 1
        }, [340, 460]);
        const decode = node('VAEDecode', '10 / Native VAE decode', [1300, 60], {}, [380, 90]);
        const save = node('SaveImage', '11 / PNG includes this workflow', [1300, 240], {
          filename_prefix: 'MOSS_T8_' + variant.toUpperCase()
        }, [430, 550]);
        node('Note', 'Read me / 使用说明', [30, 830], {
          text: 'ComfyUI official native MOSS-Image / 本地模型，不自动下载。\n选择本机已有的 DiT、moss_image TE 和匹配 VAE。\nMOSS 自带 shift=5；不要添加通用 ModelSamplingSD3。\n缓存是近似加速；enabled=false 或 threshold=0 做基线对照。\nskip22 保留；该负分支不缓存。复杂外部 patch 自动完整计算。\n图中文字与布局需自行对照。高级参数见 README。'
        }, [360, 250]);
        function connect(a, slot, b, input) {
          const index = b.findInputSlot(input);
          if (index < 0 || !a.connect(slot, b, index)) throw new Error('Failed link to ' + b.type + ':' + input);
        }
        connect(dit, 0, cache, 'model');
        connect(cache, 0, slg, 'model');
        connect(slg, 0, sampler, 'model');
        connect(clip, 0, positive, 'clip');
        connect(clip, 0, negative, 'clip');
        connect(positive, 0, sampler, 'positive');
        connect(negative, 0, sampler, 'negative');
        connect(latent, 0, sampler, 'latent_image');
        connect(sampler, 0, decode, 'samples');
        connect(vae, 0, decode, 'vae');
        connect(decode, 0, save, 'images');
        app.canvas.ds.scale = 0.84;
        app.canvas.ds.offset = [40, 40];
        return app.graph.serialize();
      }, variant);
      const file = path.join(root, 'workflows', `MOSS_T8_${variant.toUpperCase()}_1024_BlockCache.json`);
      await fs.writeFile(file, JSON.stringify(workflow, null, 2));
      await page.evaluate(() => window.app.graph.clear());
      await page.locator('#comfy-file-input').setInputFiles(file);
      await page.waitForFunction(() => window.app.graph._nodes.some(n => n.type === 'MOSSImageBlockCacheT8'));
      await page.keyboard.press('Escape');
      const state = await page.evaluate(async () => {
        const app = window.app;
        const p = await app.graphToPrompt();
        return {graph: app.graph.serialize(), api: p.output, nodes: app.graph._nodes.map(n => ({type:n.type, mode:n.mode})), missing: app.graph._nodes.filter(n => n.type?.includes('Missing')).length};
      });
      if (state.missing || state.nodes.length !== 12) throw new Error('Invalid roundtrip: ' + JSON.stringify(state));
      await fs.writeFile(file, JSON.stringify(state.graph, null, 2));
      await fs.writeFile(file.replace('.json', '-api.json'), JSON.stringify(state.api, null, 2));
      await page.screenshot({path: path.join(root, 'artifacts', `canvas-${variant}-import.png`)});
      console.log('IMPORTED', variant, JSON.stringify(state.nodes));
    }
    if (process.argv.includes('--run') || resumeId) {
      let result = {prompt_id: resumeId};
      if (!resumeId) {
      const queue = await page.evaluate(async () => (await fetch('/queue')).json());
      if (queue.queue_running.length || queue.queue_pending.length) throw new Error('Dedicated server queue is busy');
      const responsePromise = page.waitForResponse(r => r.url().endsWith('/prompt') && r.request().method() === 'POST');
      const buttons = await page.getByRole('button').allTextContents();
      console.log('BUTTONS', JSON.stringify(buttons));
      await page.getByRole('button', {name: /^(运行|Run)$/}).click();
      const response = await responsePromise;
      result = await response.json();
      if (response.status() !== 200 || !result.prompt_id) throw new Error(JSON.stringify(result));
      }
      const variant = variants.at(-1);
      await fs.writeFile(path.join(root, 'artifacts', `canvas-${variant}-submitted.json`), JSON.stringify(result, null, 2));
      console.log('CANVAS_QUEUED', result.prompt_id);
      let history;
      const deadline = Date.now() + 600000;
      while (Date.now() < deadline) {
        const h = await (await fetch(url + '/history/' + result.prompt_id)).json();
        history = h[result.prompt_id];
        if (history) break;
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
      if (!history) throw new Error('Submitted prompt is still unverified: ' + result.prompt_id);
      if (history.status?.status_str !== 'success') throw new Error(JSON.stringify(history.status));
      await fs.writeFile(path.join(root, 'artifacts', `canvas-${variant}-history.json`), JSON.stringify(history, null, 2));
      const image = Object.values(history.outputs).flatMap(v => v.images || [])[0];
      if (!image) throw new Error('Canvas run returned no saved image');
      const generated = path.join(root, 'artifacts', 'comfy-output', image.subfolder, image.filename);
      const png = path.join(root, 'workflows', `MOSS_T8_${variant.toUpperCase()}_1024_BlockCache.png`);
      await fs.copyFile(generated, png);
      await page.screenshot({path: path.join(root, 'artifacts', `canvas-${variant}-complete.png`)});
      await page.locator('#comfy-file-input').setInputFiles(png);
      await page.waitForFunction(() => window.app.graph._nodes.some(n => n.type === 'MOSSImageBlockCacheT8'));
      await page.keyboard.press('Escape');
      await page.screenshot({path: path.join(root, 'artifacts', `canvas-${variant}-png-roundtrip.png`)});
      const importedCount = await page.evaluate(() => window.app.graph._nodes.length);
      if (importedCount !== 12 || errors.length) throw new Error('PNG canvas roundtrip failed: ' + errors.join('\n'));
      if (JSON.stringify(await sourceHashes()) !== JSON.stringify(source)) throw new Error('Plugin source changed during canvas verification');
      const receiptFile = path.join(root, 'artifacts', 'canvas-acceptance.json');
      let receipts = {};
      try { receipts = JSON.parse(await fs.readFile(receiptFile, 'utf8')); }
      catch (error) { if (error.code !== 'ENOENT') throw error; }
      const pngBytes = await fs.readFile(png);
      receipts[variant] = {
        status: 'success', prompt_id: result.prompt_id, frontend_origin: 'comfyui-frontend',
        size: [pngBytes.readUInt32BE(16), pngBytes.readUInt32BE(20)], workflow_nodes: importedCount,
        png_sha256: createHash('sha256').update(pngBytes).digest('hex'), source_sha256: source,
      };
      await fs.writeFile(receiptFile, JSON.stringify(receipts, null, 2) + '\n');
      console.log('CANVAS_COMPLETE', result.prompt_id);
    }
    await fs.writeFile(path.join(root, 'artifacts', 'canvas-check.json'), JSON.stringify({errors, completed: new Date().toISOString()}, null, 2));
    if (errors.length) throw new Error('Browser errors: ' + errors.join('\n'));
  } finally {
    await browser.close();
  }
})().catch(e => {console.error(e);process.exitCode = 1;});
