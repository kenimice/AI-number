console.log("import完了");
const fs = require("fs");
const path = require("path");
const readlineSync = require("readline-sync");
const Jimp = require("jimp");
const { exec } = require('child_process');

let modecons = false;
let hidelayer = 128;
let autotimes = 1; //auto base now 800 * 3
let weight1 = 0; //in - hide [16[784]]
let weight2 = 0; //hide - out [10[16]]
let biases1 = 0; //in
let biases2 = 0; //hide
let batch = 32; //batch
let batcht = 0; //batch_count
let omomig = 0.01;   //omomi_gakusyuuritu
let baiasug = 0.01;  //baiasu_gakusyuuritu

// kari hennsuu
let correct = 0;
let times = 0;
let ans1 = 0;
let ans2 = 0;
let ansmax1 = 0;
let ansmax2 = 0;
let badcount = 0;

douki();

weight1 = JSON.parse(fs.readFileSync("weight1.json", "utf8"));
weight2 = JSON.parse(fs.readFileSync("weight2.json", "utf8"));
console.log('weight同期完了');
biases1 = JSON.parse(fs.readFileSync("biases1.json", "utf8"));
biases2 = JSON.parse(fs.readFileSync("biases2.json", "utf8"));
console.log('biases同期完了');

function softmax(arr) {
  if (modecons) console.log("softmax");
  const maxVal = Math.max(...arr);
  const expArr = arr.map((x) => Math.exp(x - maxVal));
  const sum = expArr.reduce((a, b) => a + b, 0);
  return expArr.map((x) => x / sum);
}

function leakeyrelu(x) {
  return x > 0 ? x : 0;
}

function findMaxIndex(arr) {
  const max = Math.max(...arr);
  return arr.findIndex((x) => Math.abs(x - max) < Number.EPSILON);
}

function saveWeights(file, data) {
  if (modecons) console.log("saveweights   ", file);
  fs.writeFileSync(file, JSON.stringify(data));
}

function heInit(inputSize, outputSize) {
  if (modecons) console.log("Heinit");
  const scale = Math.sqrt(2 / inputSize);
  let returnrow = [];
  for (let i = 0; i < outputSize; i++) {
      const row = [];
      for (let j = 0; j < inputSize; j++) {
          row.push((Math.random() * 2 - 1) * scale);
      }
      returnrow.push(row);
  }
  returnrow = returnrow.map((row) => row.map((v) => Math.max(-1, Math.min(v ?? 0, 1))));
  return returnrow;
}

function resetnum(input) {
if (input === 0 || input === 1) {
  weight1 = heInit(784, hidelayer);
  saveWeights("weight1.json", weight1);
}

if (input === 0 || input === 2) {
  weight2 = heInit(hidelayer, 10);
  saveWeights("weight2.json", weight2);
}

if (input === 0 || input === 3) {
  biases1 = Array.from({ length: 784 }, () => 0);
  saveWeights("biases1.json", biases1);
}

if (input === 0 || input === 4) {
  biases2 = Array.from({ length: hidelayer }, () => 0);
  saveWeights("biases2.json", biases2);
}

if (input === 0) {
  correct = 0;
  times = 0;
  ansmax1 = 0;
  ansmax2 = 0;
  badcount = 0;
}

console.log('新装開店！   ' + input);
}


async function processImage(filePath) {
  if (modecons) console.log("prosessimage");
  const img = await Jimp.read(filePath);
  img.resize(28, 28).grayscale(); // 28x28 にリサイズし、グレースケール化
  let inputArray = Array.from({ length: 28 }, () => Array(28).fill(0));
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const pixel = Jimp.intToRGBA(img.getPixelColor(x, y));
      inputArray[y][x] = pixel.r / 255; // モノクロ前提なので赤成分でOK
    }
  }
  return inputArray;
}

function flatten(input2D) {
  return input2D.flat();
}

function forward(input) {
  if (modecons) console.log("forward");
  const flatInput = flatten(input);
  let hidden = Array(hidelayer).fill(0);
  for (let i = 0; i < 784; i++) {
    flatInput[i] += biases1[i];
  }

  for (let i = 0; i < hidelayer; i++) {
    for (let j = 0; j < 784; j++) {
      hidden[i] += (flatInput[j] * weight1[i][j]) / 784;
    }
    hidden[i] += biases2[i];
    hidden[i] = leakeyrelu(hidden[i]);
  }

  hidden.forEach((value, index) => {
    if (typeof value !== "number" || isNaN(value)) {
      console.log(`arr[${index}] is not a number:`, value);
      value = 0;
    }
  });

  let output = Array(10).fill(0);
  for (let i = 0; i < 10; i++) {
    for (let j = 0; j < hidelayer; j++) {
      output[i] += (hidden[j] * weight2[i][j]) / hidelayer;
    }
  }
  //output is big
  output = softmax(output);
  //softmax all 1> >0
  return {
    output,
    hidden,
    flatInput,
  };
}
let sa1 = []; //hide - out weight2 [10[16]]
let sa2 = []; //in - hide weight1 [16[784]]
let sa3 = []; //hide biases2 16
let sa4 = []; //in biases1 784
function backpropagate(flatInput, hidden, output, label) {
  if (modecons) console.log("backpropagate");
  let cost = Array(10).fill(0);
  cost = output.map((o, i) => o - (i === label ? 1 : 0));
  sa1 = Array.from({ length: 10 }, () => Array(hidelayer).fill(0));
  sa2 = Array.from({ length: hidelayer }, () => Array(784).fill(0));
  sa3 = Array(hidelayer).fill(0);
  sa4 = Array(784).fill(0);
  sa1 = sa1.map(row => row.map(v => parseFloat(v)));
  sa2 = sa2.map(row => row.map(v => parseFloat(v)));
  sa3 = sa3.map(v => parseFloat(v));
  sa4 = sa4.map(v => parseFloat(v));
  for (let i = 0; i < 10; i++) {
    if (i === label) {
      cost[i] = output[i] ** 2;
    }else {
      cost[i] = (output[i] - 1) ** 2;
    }
  }
  for (let i = 0; i < 10; i++) {
    for (let j = 0; j < hidelayer; j++) {
      sa1[i][j] -= (cost[i] * weight2[i][j]);
      sa3[j] -= (cost[i] * weight2[i][j]) / hidelayer;
    }
  }
  for (let i = 0; i < hidelayer; i++) {
    for (let j = 0; j < 784; j++) {
      for (let k = 0; k < 10; k++) {
        sa2[i][j] -= (sa1[k][i] * weight1[i][j]) / 10;
        sa4[j] -= (sa3[j] * weight1[i][j]) / hidelayer;
      }
    }
  }
  batcht ++;
  if (batcht >= batch) {
    batcht = 0;
    if (learn) kaisei(flatInput);
  }
  cost.forEach((value, index) => {
    if (typeof value !== "number" || isNaN(value)) {
      console.log(`cost[${index}] is not a number:`);
      value = 0;
    }
  });
}

function kaisei(flatInput){
  if (modecons) console.log("kaisei");
  for (let i = 0; i < hidelayer; i++) {
    for (let j = 0; j < 10; j++) {
      weight2[j][i] -= sa1[j][i] * omomig;
    }

    biases2[i] -= sa3[i] * baiasug;
  }
  weight2 = weight2.map((row) => row.map((v) => Math.max(-1, Math.min(v, 1))));
  biases2 = biases2.map((v) => Math.max(-1, Math.min(v ?? 0, 1)));

  for (let i = 0; i < hidelayer; i++) {
    for (let j = 0; j < 784; j++) {
      weight1[i][j] -= sa2[i][j] * omomig;
    }
    biases1[i] -= sa4[i] * baiasug;
  }

  weight1 = weight1.map((row) => row.map((v) => Math.max(-1, Math.min(v, 1))));
  biases1 = biases1.map((v) => Math.max(-1, Math.min(v ?? 0, 1)));
}

function douki() {
  if (modecons) console.log("douki");
  const fff = ["weight1", "weight2", "biases1", "biases2"];
  let allExist = true;
  fff.forEach((value) => {
    if (!fs.existsSync(`${value}.json`)) {
      fs.writeFileSync(`${value}.json`, "");
      allExist = false;
    }
  });

  if (allExist) {
    weight1 = JSON.parse(fs.readFileSync("weight1.json", "utf8"));
    weight2 = JSON.parse(fs.readFileSync("weight2.json", "utf8"));
    biases1 = JSON.parse(fs.readFileSync("biases1.json", "utf8"));
    biases2 = JSON.parse(fs.readFileSync("biases2.json", "utf8"));
  }
  
  if (fs.readFileSync("weight1.json", "utf8") == 0) resetnum(1);
  if (fs.readFileSync("weight2.json", "utf8") == 0) resetnum(2);
  if (fs.readFileSync("biases1.json", "utf8") == 0) resetnum(3);
  if (fs.readFileSync("biases2.json", "utf8") == 0) resetnum(4);
}

async function runTrainingLoop(imageFolderPath, auto) {
  if (modecons) console.log("runtrainingroop");
  const files = fs
    .readdirSync(imageFolderPath)
    .filter((f) => f.endsWith(".png"));
  for (const file of files) {
    const fullPath = path.join(imageFolderPath, file);
    const match = file.match(/-num(\d+)\.png$/);
    if (!match) {
      console.log(`❌ 無視: ${file}`);
      continue;
    }
    const label = parseInt(match[1], 10);
    const input = await processImage(fullPath);
    const { output, hidden, flatInput } = forward(input);

    const predicted = findMaxIndex(output);
    if (predicted === label) correct++;
    
    //事後処理
    times ++;
    if (times > 200 && parseFloat(((correct / times) * 100).toFixed(2)) > parseFloat(ansmax1)) {
      ansmax1 = parseFloat(((correct / times) * 100).toFixed(2));
      ansmax2 = times;
    }
    ans2 = ans1;
    ans1 = ((correct / times) * 100).toFixed(2);
    if (!auto || !quiet) {
      if (parseFloat(ans1) < parseFloat(ans2)) {
        if (predicted === label) console.log(`善 ${file} → 予測: ${predicted}, 正解: ${label}, 正答率: ${ans1}%, 試行回数： ${times} bad`);
        else console.log(`悪 ${file} → 予測: ${predicted}, 正解: ${label}, 正答率: ${ans1}%, 試行回数： ${times} bad`);
        badcount++;
      }else {
        if (predicted === label) console.log(`善 ${file} → 予測: ${predicted}, 正解: ${label}, 正答率: ${ans1}%, 試行回数： ${times}`);
        else console.log(`悪 ${file} → 予測: ${predicted}, 正解: ${label}, 正答率: ${ans1}%, 試行回数： ${times}`);
      }
      quiet = false;
    }
    const logfile = path.join(__dirname, 'log2.txt');
    fs.appendFileSync(logfile, ans1 + "\n");
    if (times % 10000 === 0) {
      baiasug = baiasug / 2;
      omomig = omomig / 2;
      console.log(`b, bnを0.001減らしました ${times}回 b= ${omomig}`);
    }
    //学習
    backpropagate(flatInput, hidden, output, label);
  }
  
  
  saveWeights("weight1.json", weight1);
  saveWeights("weight2.json", weight2);
  saveWeights("biases1.json", biases1);
  saveWeights("biases2.json", biases2);
}

function saveNetworkToFolder() {
  let folderName = readlineSync.question('ファイル名：');
  const dirPath = path.join(__dirname, folderName);
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath);
    console.log(`📁 フォルダ作成: ${folderName}`);
  }
  
  let hennsuus = {weight1, weight2, biases1, biases2};

  for (const [name, value] of Object.entries(hennsuus)) {
    const filePath = path.join(dirPath, `${name}.json`);
    const content = `${name}\omomig${JSON.stringify(value)}`;
    fs.writeFileSync(filePath, content, "utf-8");
    console.log(`✅ 保存完了: ${name}.json`);
  }

}

function outlog(count) {
    const logfile = path.join(__dirname, 'log.txt');
    if (count === 1) {
      fs.appendFileSync(logfile, `\omomig${count}st: ${((correct / times) * 100).toFixed(2)}%, maxtime: ${ansmax2}t, max: ${ansmax1}% badcount: ${badcount}\omomig`, 'utf-8');
    }else if (count === 2) {
      fs.appendFileSync(logfile, `${count}nd: ${((correct / times) * 100).toFixed(2)}%, maxtime: ${ansmax2}t, max: ${ansmax1}% badcount: ${badcount}\omomig`, 'utf-8');
    }else if (count === 3) {
      fs.appendFileSync(logfile, `${count}rd: ${((correct / times) * 100).toFixed(2)}%, maxtime: ${ansmax2}t, max: ${ansmax1}% badcount: ${badcount}\omomig`, 'utf-8');
    }else {
      fs.appendFileSync(logfile, `${count}th: ${((correct / times) * 100).toFixed(2)}%, maxtime: ${ansmax2}t, max: ${ansmax1}% badcount: ${badcount}\omomig`, 'utf-8');
    }
}

let auto = false;
let autocount = 1;
let quiet = false;
let learn = true;
async function foldera(on) {
  if (modecons) console.log("foldera");
  if (on) quiet = true;
  if (on) auto = true;
  else auto = false;
  const currentDir = __dirname;
  const files = fs.readdirSync(currentDir);
  const testCluFiles = files.filter((file) => file.startsWith("test_clu"));
  douki();
  for (let file of testCluFiles) {
    file = __dirname + "/" + file;
    await runTrainingLoop(file, auto);
  }
  if (on && ((correct / times) * 100).toFixed(2) < 12.00) {
    console.log(((correct / times) * 100).toFixed(2) + '%');
    resetnum(0);
    setTimeout(() => {
      foldera(1);
    }, 500);
  }else {
    if (on) console.log(((correct / times) * 100).toFixed(2) + '%');
    console.log(`max: ${ansmax1}%, times: ${ansmax2} badcount: ${badcount}`);
    outlog(autocount);
    if (autocount < autotimes) {
      quiet = true;
      autocount++;
      douki();
      foldera(0);
      console.log('start again');
    }else{
      autocount = 1;
      exec('.bat', (err, stdout, stderr) => {
        if (err) {
          console.error(err);
          return;
        }
        console.log(stdout);
      });
      start().catch(err => console.error(err));
    }
  }
}

async function start() {
  console.log("start");
  let folder = readlineSync.question("選択してくださいc/i/a/b/s/u/l: ");
  if (folder === "c") {
    folder = readlineSync.question("ほんとに？y/n: ");
    if (folder === "y") {
      resetnum(0);
    }
    start();
  } else if (folder === "i") {
    folder = readlineSync.question("画像フォルダのパスを入力してください: ");
    douki();
    runTrainingLoop(folder);
  } else if (folder === "a") {
    douki();
    foldera(0);
  }else if (folder === "ca") {
    douki();
    quiet = true;
    foldera(0);
  }else if (/^a\*\d+$/.test(folder)) {
    douki();
    let split = parseInt(folder.split("*")[1]);
    for (let i = 0; i < split; i++) {
      const currentDir = __dirname;
      const files = fs.readdirSync(currentDir);
      const testCluFiles = files.filter((file) => file.startsWith("test_clu"));
      for (let file of testCluFiles) {
        file = __dirname + "/" + file;
        await runTrainingLoop(file);
      }
    }
    start().catch(err => console.error(err));
  }else if (folder === "b") {
    console.log(`今： ${omomig}, ${baiasug}`);
    folder = readlineSync.question("値: ");
    if (folder !== "") {
      omomig = parseFloat(folder);
      baiasug = parseFloat(folder);
    }
    console.log(`バイアス：${omomig} ${baiasug}`);
    start();
  }else if (folder === 's') {
    saveNetworkToFolder();
    start();
  }else if (folder === 'u') {
    resetnum(0);
    foldera(1);
  }else if (folder === 'l'){
    console.log(`now: ${learn}`);
    if (learn) learn = false;
    else learn = true;
    console.log(`change: ${learn}`);
    start();
  }else if (folder === 'help') {
    console.log(`c: clear データ全消し\ni: custom カスタムファイル\na: all file 存在するすべての画像を見る\nb: bias 学習率変更 今:${omomig}\ns: save データを同じ階層に保存\nu: auto 800時点で11%以上のデータのみ通過\nl: learn 現在のデータを読み込むときに学習するかしないか`);
  }else {
    console.log('error');
    start();
  }
}

start();
