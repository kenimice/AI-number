const grid = document.getElementById("grid");

for (let i = 0; i < 28 * 28; i++) {
    const cell = document.createElement("div");
    cell.classList.add("cell");
    const row = Math.floor(i / 28) + 1;
    const col = (i % 28) + 1;
    cell.id = `cell-${row}-${col}`;
    cell.style.backgroundColor = `rgba(0,0,0,1)`;
    grid.appendChild(cell);
}

document.querySelectorAll(".cell").forEach(cell => {
    cell.setAttribute("draggable", "false");
});

let isDragging = false;
let latestPointer = null;
let animationFrameId = null;

// マウスイベント
document.addEventListener("mousedown", (e) => {
    isDragging = true;
    handlePointerMove(e.clientX, e.clientY);
});

document.addEventListener("mousemove", (e) => {
    if (isDragging) {
        handlePointerMove(e.clientX, e.clientY);
    }
});

document.addEventListener("mouseup", () => {
    isDragging = false;
});

// タッチイベント
document.addEventListener("touchstart", (e) => {
    isDragging = true;
    e.preventDefault(); // タッチスクロール防止
    let touch = e.touches[0];
    handlePointerMove(touch.clientX, touch.clientY);
}, { passive: false });

document.addEventListener("touchmove", (e) => {
    if (isDragging) {
        e.preventDefault(); // タッチスクロール防止
        let touch = e.touches[0];
        latestPointer = { x: touch.clientX, y: touch.clientY };

        // `requestAnimationFrame()` で描画負荷を下げる
        if (!animationFrameId) {
            animationFrameId = requestAnimationFrame(() => {
                handlePointerMove(latestPointer.x, latestPointer.y);
                animationFrameId = null;
            });
        }
    }
}, { passive: false });

document.addEventListener("touchend", () => {
    isDragging = false;
});

// 座標を処理する関数
function handlePointerMove(x, y) {
    color(x, y);
}

function alle() {
for (let i = 1; i < 29; i++) {
    for (let j = 1; j < 29; j++) {
            let cll = document.getElementById(`cell-${j}-${i}`);
            cll.style.backgroundColor = 'rgba(0,0,0,1)';
        }
    }
}

document.getElementById('enter').addEventListener('touchend', () => {
    enter();
});

document.getElementById('easports').addEventListener('touchend', () => {
    alle();
});

function digits(number) {
    return (number + 800).toString().padStart(6, '0');
}

function gen(length) {
    let result = [];
    while (result.length < length) {
        let set = new Set();
        while (set.size < 10) {
            set.add(Math.floor(Math.random() * 10));
        }
        result.push(...set);
    }
    return result.slice(0, length);
}

let num = gen(100);
console.log("a        " + num);

function color(x,y) {
    width = window.innerWidth;
    left = width / 2 - 280;
    for (let i = 1; i < 29; i++) {
        for (let j = 1; j < 29; j++) {
            let l, c, a;
            l = left + 9.3 + (i - 1) * 18.7;
            c = 58 + 9.3 + (j - 1) * 18.7;
            a = Math.sqrt((x - l) ** 2 + (y - c) ** 2);
            if (30 - a > 0) {
                let cll = document.getElementById(`cell-${j}-${i}`);
                let now_color = window.getComputedStyle(cll).backgroundColor;
                let rgbaMatch = now_color.match(/^rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)$/);
                let alpha = 1;
                if (rgbaMatch) {
                    alpha = rgbaMatch[4] !== undefined ? parseFloat(rgbaMatch[4]) : 1;
                }
                a = a / 30;
                if (alpha > a) {
                    cll.style.backgroundColor = `rgba(0, 0, 0, ${a})`;
                }
            }
        }
    }    
}

let time = 0;
document.getElementById('num').innerHTML = num[time];

function enter() {
        time++;
        time = time % 100;
        document.getElementById('num').innerHTML = num[time];
        let canvas = document.createElement("canvas");
        canvas.width = 28;
        canvas.height = 28;
        let ctx = canvas.getContext("2d");
        // すべてのセルの色を取り出して、キャンバスに描画
        document.querySelectorAll(".cell").forEach((cell, index) => {
            let row = Math.floor(index / 28); // 行番号
            let col = index % 28; // 列番号
            let color = window.getComputedStyle(cell).backgroundColor;
            let rgba = parseFloat(color.match(/\d+/g)[4] ?? 1000);
            while (rgba < 100 && rgba !== 0) {
                rgba = rgba * 10;
            }
            let nou = ((1000 - rgba) / 1000).toFixed(5) * 255;
            ctx.fillStyle = `rgb(${nou},${nou},${nou})`;
            ctx.fillRect(col, row, 1, 1); // キャンバスに1pxのドットを描画
        });
        // 画像データをBase64形式で取得
        const imageData = canvas.toDataURL("image/png");
        // ダウンロード用のリンクを作成
        const link = document.createElement("a");
        link.href = imageData;
        link.download = `${digits(time - 1)}-num${num[time - 1]}.png`;
        link.click();
        alle();
        if (time % 10 === 0) {
            console.log(time);
        }
}