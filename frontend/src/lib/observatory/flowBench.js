/* The Flow Bench engine (Phase 6.1).

   Inference rendered as liquid. Every motion traces to a real lifecycle event
   from lib/telemetryLog's bus — the engine has no random event generator and
   nothing injects without a request. Idle means the field settles; stillness
   is honest information.

   Renderers: WebGL dye advection under a curl-noise velocity field
   (divergence-free by construction — the "curl-noise advection class"), with a
   Canvas2D particle-advection fallback sharing the same request choreography.

   Color discipline: colors are read from tokens.css custom properties at init
   and on theme change. Prompt ink = steel blue (--color-accent), generation
   ink = neutral warm grey-white (--color-text), error bloom = --color-error at
   low saturation. Copper and amber are claim colors and never appear here. */

const MAX_TRACE_POINTS = 240

function cssColor(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return value || fallback
}

function parseColor(value) {
  const probe = document.createElement('canvas').getContext('2d')
  probe.fillStyle = value
  const normalized = probe.fillStyle
  if (normalized.startsWith('#')) {
    const n = parseInt(normalized.slice(1), 16)
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]
  }
  const m = normalized.match(/rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)/)
  return m ? [m[1] / 255, m[2] / 255, m[3] / 255] : [0.5, 0.6, 0.7]
}

function desaturate([r, g, b], amount) {
  const grey = 0.3 * r + 0.6 * g + 0.1 * b
  return [r + (grey - r) * amount, g + (grey - g) * amount, b + (grey - b) * amount]
}

export function readPalette() {
  return {
    prompt: parseColor(cssColor('--color-accent', '#8fb6dc')),
    generation: desaturate(parseColor(cssColor('--color-text', '#dde5ed')), 0.15),
    error: desaturate(parseColor(cssColor('--color-error', '#e9928a')), 0.45),
  }
}

/* ---- Request choreography (shared by both renderers) ---- */
export function createChoreography() {
  const active = new Map()
  const traces = new Map() // id -> [{x,y}] in 0..1 field space
  let inletSlot = 0
  return {
    traces,
    active,
    start(event) {
      inletSlot = (inletSlot + 1) % 5
      const y = 0.22 + inletSlot * 0.14
      const req = {
        id: event.id,
        kind: event.kind,
        y,
        x: 0.04,
        phase: 'drift',     // drifting prompt ink until first content
        tokensPerSec: 0,
        startedAt: performance.now(),
        endedAt: null,
        outcome: null,
      }
      active.set(event.id, req)
      traces.set(event.id, [{ x: req.x, y }])
      if (traces.size > 24) traces.delete(traces.keys().next().value)
      return req
    },
    firstContent(event) {
      const req = active.get(event.id)
      if (req) req.phase = 'burst' // engine consumes one burst then sets 'flow'
      return req
    },
    progress(event) {
      const req = active.get(event.id)
      if (req && Number.isFinite(event.tokensPerSec)) req.tokensPerSec = event.tokensPerSec
      return req
    },
    end(event) {
      const req = active.get(event.id)
      if (req) {
        req.phase = event.outcome === 'ok' ? 'mixing' : event.outcome === 'interrupted' ? 'cut' : 'bloom'
        req.outcome = event.outcome
        req.endedAt = performance.now()
        // engine consumes the terminal phase, then removes after its gesture
      }
      return req
    },
    trace(id, x, y) {
      const list = traces.get(id)
      if (list && list.length < MAX_TRACE_POINTS) list.push({ x, y })
    },
    prune() {
      for (const [id, req] of active) {
        if (req.endedAt && performance.now() - req.endedAt > 4000) active.delete(id)
      }
    },
  }
}

/* ---- WebGL renderer ---- */
const VERT = `attribute vec2 p;varying vec2 uv;void main(){uv=p*0.5+0.5;gl_Position=vec4(p,0.,1.);}`

const NOISE = `
vec2 hash(vec2 p){p=vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3)));return -1.+2.*fract(sin(p)*43758.5453123);}
float noise(vec2 p){vec2 i=floor(p),f=fract(p);vec2 u=f*f*(3.-2.*f);
return mix(mix(dot(hash(i),f),dot(hash(i+vec2(1.,0.)),f-vec2(1.,0.)),u.x),
mix(dot(hash(i+vec2(0.,1.)),f-vec2(0.,1.)),dot(hash(i+vec2(1.,1.)),f-vec2(1.,1.)),u.x),u.y);}
vec2 curl(vec2 p,float t){float e=0.08;
float n1=noise(p+vec2(0.,e)+t*0.06),n2=noise(p-vec2(0.,e)+t*0.06);
float n3=noise(p+vec2(e,0.)+t*0.06),n4=noise(p-vec2(e,0.)+t*0.06);
return vec2((n1-n2)/(2.*e),-(n3-n4)/(2.*e));}`

const ADVECT = `precision highp float;varying vec2 uv;uniform sampler2D dye;
uniform float t;uniform float dt;uniform float ambient;uniform vec4 jets[8];uniform int jetCount;
${NOISE}
void main(){
vec2 vel=curl(uv*3.0,t)*ambient;
vel.x+=ambient*0.6; /* gentle left-to-right bench current */
for(int i=0;i<8;i++){if(i>=jetCount)break;
vec2 jp=jets[i].xy;float power=jets[i].z;
vec2 d=uv-jp;float fall=exp(-dot(d,d)*220.0);
vel+=vec2(power,0.18*sin(t*2.0+jp.y*40.0))*fall;}
vec2 src=uv-vel*dt;
vec4 c=texture2D(dye,src);
gl_FragColor=c*0.988; /* slow diffusion toward ambient */
}`

const SPLAT = `precision highp float;varying vec2 uv;uniform sampler2D dye;
uniform vec2 point;uniform vec3 color;uniform float radius;
void main(){vec2 d=uv-point;d.x*=1.6;float a=exp(-dot(d,d)/radius);
vec4 base=texture2D(dye,uv);gl_FragColor=vec4(base.rgb+color*a,1.0);}`

const DISPLAY = `precision highp float;varying vec2 uv;uniform sampler2D dye;
void main(){vec3 c=texture2D(dye,uv).rgb;
c=c*1.9/(1.0+c); /* soft tonemap: luminous ink, never blown white */
float lum=dot(c,vec3(0.55));
gl_FragColor=vec4(c,clamp(lum*2.6,0.0,0.92));}`

function compile(gl, type, src) {
  const sh = gl.createShader(type)
  gl.shaderSource(sh, src)
  gl.compileShader(sh)
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh))
  return sh
}

function program(gl, frag) {
  const pr = gl.createProgram()
  gl.attachShader(pr, compile(gl, gl.VERTEX_SHADER, VERT))
  gl.attachShader(pr, compile(gl, gl.FRAGMENT_SHADER, frag))
  gl.linkProgram(pr)
  if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(pr))
  return pr
}

export function createWebGLBench(canvas, simScale = 0.5) {
  const gl = canvas.getContext('webgl', { alpha: true, antialias: false, preserveDrawingBuffer: true })
  if (!gl || gl.isContextLost()) return null
  const half = gl.getExtension('OES_texture_half_float')
  gl.getExtension('OES_texture_half_float_linear')
  const texType = half ? half.HALF_FLOAT_OES : gl.UNSIGNED_BYTE

  const quad = gl.createBuffer()
  gl.bindBuffer(gl.ARRAY_BUFFER, quad)
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW)

  const advectPr = program(gl, ADVECT)
  const splatPr = program(gl, SPLAT)
  const displayPr = program(gl, DISPLAY)

  let simW = 0
  let simH = 0
  let fbos = null

  function makeTarget(w, h) {
    const tex = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, tex)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, texType, null)
    const fbo = gl.createFramebuffer()
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0)
    return { tex, fbo }
  }

  function resize(width, height) {
    canvas.width = width
    canvas.height = height
    simW = Math.max(64, Math.round(width * simScale))
    simH = Math.max(64, Math.round(height * simScale))
    fbos = [makeTarget(simW, simH), makeTarget(simW, simH)]
  }

  function draw(pr) {
    gl.useProgram(pr)
    const loc = gl.getAttribLocation(pr, 'p')
    gl.bindBuffer(gl.ARRAY_BUFFER, quad)
    gl.enableVertexAttribArray(loc)
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0)
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
  }

  return {
    kind: 'webgl',
    resize,
    splat(x, y, color, radius) {
      gl.viewport(0, 0, simW, simH)
      gl.useProgram(splatPr)
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbos[1].fbo)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, fbos[0].tex)
      gl.uniform1i(gl.getUniformLocation(splatPr, 'dye'), 0)
      gl.uniform2f(gl.getUniformLocation(splatPr, 'point'), x, 1 - y)
      gl.uniform3f(gl.getUniformLocation(splatPr, 'color'), ...color)
      gl.uniform1f(gl.getUniformLocation(splatPr, 'radius'), radius)
      draw(splatPr)
      fbos.reverse()
    },
    step(t, dt, ambient, jets) {
      gl.viewport(0, 0, simW, simH)
      gl.useProgram(advectPr)
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbos[1].fbo)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, fbos[0].tex)
      gl.uniform1i(gl.getUniformLocation(advectPr, 'dye'), 0)
      gl.uniform1f(gl.getUniformLocation(advectPr, 't'), t)
      gl.uniform1f(gl.getUniformLocation(advectPr, 'dt'), dt)
      gl.uniform1f(gl.getUniformLocation(advectPr, 'ambient'), ambient)
      const flat = new Float32Array(32)
      jets.slice(0, 8).forEach((jet, i) => {
        flat[i * 4] = jet.x
        flat[i * 4 + 1] = 1 - jet.y
        flat[i * 4 + 2] = jet.power
        flat[i * 4 + 3] = 0
      })
      gl.uniform4fv(gl.getUniformLocation(advectPr, 'jets'), flat)
      gl.uniform1i(gl.getUniformLocation(advectPr, 'jetCount'), Math.min(jets.length, 8))
      draw(advectPr)
      fbos.reverse()
    },
    render() {
      gl.viewport(0, 0, canvas.width, canvas.height)
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      gl.clearColor(0, 0, 0, 0)
      gl.clear(gl.COLOR_BUFFER_BIT)
      gl.enable(gl.BLEND)
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
      gl.useProgram(displayPr)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, fbos[0].tex)
      gl.uniform1i(gl.getUniformLocation(displayPr, 'dye'), 0)
      draw(displayPr)
    },
    destroy() {
      /* Do NOT loseContext() here: getContext on the same canvas returns the
         same context object, so a deliberate loss bricks the next mount
         (React strict-mode double-mount hit exactly this). GC handles it. */
    },
  }
}

/* ---- Canvas2D particle fallback ---- */
function jsNoise(x, y) {
  return Math.sin(x * 2.1 + Math.cos(y * 1.7)) * Math.cos(y * 1.9 - Math.sin(x * 1.3))
}

function jsCurl(x, y, t) {
  const e = 0.08
  const tx = t * 0.06
  return [
    (jsNoise(x + tx, y + e) - jsNoise(x + tx, y - e)) / (2 * e),
    -(jsNoise(x + e + tx, y) - jsNoise(x - e + tx, y)) / (2 * e),
  ]
}

export function createCanvas2DBench(canvas, maxParticles = 900) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  const particles = []
  return {
    kind: 'canvas2d',
    resize(width, height) {
      canvas.width = width
      canvas.height = height
    },
    splat(x, y, color, radius) {
      const count = Math.min(26, Math.round(radius * 9000))
      for (let i = 0; i < count; i++) {
        if (particles.length >= maxParticles) particles.shift()
        particles.push({
          x: x + (Math.random() - 0.5) * radius * 18,
          y: y + (Math.random() - 0.5) * radius * 18,
          color,
          life: 1,
        })
      }
    },
    step(t, dt, ambient, jets) {
      for (const particle of particles) {
        const [vx, vy] = jsCurl(particle.x * 3, particle.y * 3, t)
        let dx = (vx * ambient + ambient * 0.6) * dt
        let dy = vy * ambient * dt
        for (const jet of jets) {
          const ddx = particle.x - jet.x
          const ddy = particle.y - jet.y
          const fall = Math.exp(-(ddx * ddx + ddy * ddy) * 220)
          dx += jet.power * fall * dt
        }
        particle.x += dx
        particle.y += dy
        particle.life *= 0.992
      }
      while (particles.length && particles[0].life < 0.04) particles.shift()
    },
    render() {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      for (const particle of particles) {
        const [r, g, b] = particle.color
        ctx.fillStyle = `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},${(particle.life * 0.5).toFixed(3)})`
        ctx.beginPath()
        ctx.arc(particle.x * canvas.width, particle.y * canvas.height, Math.max(1.5, canvas.width * 0.004 * particle.life), 0, Math.PI * 2)
        ctx.fill()
      }
    },
    destroy() {},
  }
}

export function createBench(canvas, { preferWebGL = true } = {}) {
  if (preferWebGL) {
    try {
      const bench = createWebGLBench(canvas)
      if (bench) return bench
    } catch { /* fall through to 2D */ }
  }
  return createCanvas2DBench(canvas)
}
