/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as posedetection from '@tensorflow-models/pose-detection';
import * as scatter from 'scatter-gl';
import { runInThisContext } from 'vm';

import * as params from './params';
import {isMobile} from './util';
import {noise} from '../noisejs/perlin.js';

var Victor = require('victor');

// These anchor points allow the pose pointcloud to resize according to its
// position in the input.
const ANCHOR_POINTS = [[0, 0, 0], [0, 1, 0], [-1, 0, 0], [-1, -1, 0]];

// #ffffff - White
// #800000 - Maroon
// #469990 - Malachite
// #e6194b - Crimson
// #42d4f4 - Picton Blue
// #fabed4 - Cupid
// #aaffc3 - MvarGreen
// #9a6324 - Kumera
// #000075 - Navy Blue
// #f58231 - Jaffa
// #4363d8 - Royal Blue
// #ffd8b1 - Caramel
// #dcbeff - Mauve
// #808000 - Olive
// #ffe119 - Candlelight
// #911eb4 - Seance
// #bfef45 - Inchworm
// #f032e6 - Razzle Dazzle Rose
// #3cb44b - Chateau Green
// #a9a9a9 - Silver Chalice
const COLOR_PALETTE = [
  '#ffffff', '#800000', '#469990', '#e6194b', '#42d4f4', '#fabed4', '#aaffc3',
  '#9a6324', '#000075', '#f58231', '#4363d8', '#ffd8b1', '#dcbeff', '#808000',
  '#ffe119', '#911eb4', '#bfef45', '#f032e6', '#3cb44b', '#a9a9a9'
];

function hslToRgb(h, s, l){
  var r, g, b;

  if(s == 0){
      r = g = b = l; // achromatic
  }else{
      var hue2rgb = function hue2rgb(p, q, t){
          if(t < 0) t += 1;
          if(t > 1) t -= 1;
          if(t < 1/6) return p + (q - p) * 6 * t;
          if(t < 1/2) return q;
          if(t < 2/3) return p + (q - p) * (2/3 - t) * 6;
          return p;
      }

      var q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      var p = 2 * l - q;
      r = hue2rgb(p, q, h + 1/3);
      g = hue2rgb(p, q, h);
      b = hue2rgb(p, q, h - 1/3);
  }

  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

class Ball {
  constructor() {
    this.pos = new Victor(0,0);
    this.vel = new Victor(0,0);
    this.acc = new Victor(0,0);
    this.target = null;
    this.rand = (Math.random() - 0.5) / 10;

    this.canvas = document.getElementById('output');
    this.ctx = this.canvas.getContext('2d');
  }

  update(target) {
    /* control ball movement */
    var damp = target.pos.clone().subtract(this.pos).length();
    if (damp / 300 < 1) {
       damp = Math.min(damp / 100, 1);
    }

    else {
      damp = 1;
    }
    damp = Math.max(damp, 0);

    this.acc = target.pos.clone().subtract(this.pos).normalize().multiply(new Victor(1.5 + this.rand, 1.5 + this.rand)).add(new Victor(this.rand / 2, this.rand / 2));
    this.vel = this.vel.clone().add(this.acc);
    this.vel = this.vel.clone().multiply(new Victor(damp, damp));
    var curr_vel = this.vel.length();
    var new_vel = Math.min(curr_vel, 50);
    this.vel = this.vel.clone().normalize().multiply(new Victor(new_vel, new_vel));
    this.vel = this.vel.multiply(new Victor(0.98, 0.98));
    this.pos = this.pos.clone().add(this.vel);
  }
}

class Finger {
  constructor() {
    this.pos = new Victor(0,0);
    this.vel = new Victor(0,0);
    this.acc = new Victor(0,0);
    this.target = null;
    this.pastAcc = [];
    this.avgAcc = 0;
  }

  update(pose) {
    var new_pos = new Victor(pose.keypoints[9].x, pose.keypoints[9].y);
    var new_vel = new_pos.clone().subtract(this.pos);
    this.pos = new_pos;
    this.acc = new_vel.clone().subtract(this.vel);
    this.vel = new_vel;

    this.pastAcc.push(this.acc.length());
    if (this.pastAcc.length > 5) {
      this.pastAcc.shift();
    }

    var tmp = 0
    for (let i = 0; i < this.pastAcc.length; i++) {
      tmp += this.pastAcc[i];
    }
    this.avgAcc = tmp / this.pastAcc.length;
  }
}

export class Camera {
  constructor() {
    this.video = document.getElementById('video');
    this.canvas = document.getElementById('output');

    this.ctx = this.canvas.getContext('2d');
    this.scatterGLEl = document.querySelector('#scatter-gl-container');
    this.scatterGL = new scatter.ScatterGL(this.scatterGLEl, {
      'rotateOnStart': true,
      'selectEnabled': false,
      'styles': {polyline: {defaultOpacity: 1, deselectedOpacity: 1}}
    });
    this.scatterGLHasInitialized = false;
    this.pastPoses = [];
    this.maxPoses = 50;
    this.speed = 0.5;
    this.falloff = 5;
    this.falloffEdge = 2;
    this.falloffOrb = 5;
    this.color = "200, 200, 200";
    this.color2 = "255, 255, 255";
    this.time = 0;

    this.balls = []
    // for (var i = 0; i < 5; i++) {
    //   this.balls.push(new Ball());
    // }
    this.finger = new Finger();

    this.canvas.addEventListener("click",this.fullscreen) 
  }

  fullscreen(){
    var canvas = document.getElementById('output');
    if(canvas.webkitRequestFullScreen) {
      canvas.webkitRequestFullScreen();
    }
    else {
      canvas.mozRequestFullScreen();
    }            
  }

  /**
   * Initiate a Camera instance and wait for the camera stream to be ready.
   * @param cameraParam From app `STATE.camera`.
   */
  static async setupCamera(cameraParam) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
          'Browser API navigator.mediaDevices.getUserMedia not available');
    }

    const {targetFPS, sizeOption} = cameraParam;
    const $size = params.VIDEO_SIZE[sizeOption];
    const videoConfig = {
      'audio': false,
      'video': {
        facingMode: 'user',
        // Only setting the video to a specified size for large screen, on
        // mobile devices accept the default size.
        width: isMobile() ? params.VIDEO_SIZE['360 X 270'].width : $size.width,
        height: isMobile() ? params.VIDEO_SIZE['360 X 270'].height :
                             $size.height,
        frameRate: {
          ideal: targetFPS,
        }
      }
    };

    const stream = await navigator.mediaDevices.getUserMedia(videoConfig);

    const camera = new Camera();
    camera.video.srcObject = stream;

    await new Promise((resolve) => {
      camera.video.onloadedmetadata = () => {
        resolve(video);
      };
    });

    camera.video.play();

    const videoWidth = camera.video.videoWidth;
    const videoHeight = camera.video.videoHeight;
    // Must set below two lines, otherwise video element doesn't show.
    camera.video.width = videoWidth;
    camera.video.height = videoHeight;

    camera.canvas.width = videoWidth;
    camera.canvas.height = videoHeight;
    const canvasContainer = document.querySelector('.canvas-wrapper');
    canvasContainer.style = `width: ${videoWidth}px; height: ${videoHeight}px`;

    // Because the image from camera is mirrored, need to flip horizontally.
    camera.ctx.translate(camera.video.videoWidth, 0);
    camera.ctx.scale(-1, 1);

    camera.scatterGLEl.style =
        `width: ${videoWidth}px; height: ${videoHeight}px;`;
    camera.scatterGL.resize();

    camera.scatterGLEl.style.display =
        params.STATE.modelConfig.render3D ? 'inline-block' : 'none';

    return camera;
  }

  drawCtx() {
    this.ctx.draw
    this.ctx.beginPath();
    // this.ctx.fillStyle = "rgba(0,0,0, 0.05)";
    if (this.to_invert) {
      this.ctx.fillStyle = "rgb(255,255,255)";

      this.to_invert = false;
    }
    else {
      this.ctx.fillStyle = "rgb(0,0,0)";
    }
    this.ctx.rect(0, 0, this.video.videoWidth, this.video.videoHeight);
    this.ctx.stroke();
    this.ctx.fill();

    this.time += 1;
    if (this.speed_up > 20) {
      this.speed_up -= 90
      this.speed_up = this.speed_up / 110 * 1.5
      this.time += this.speed_up;
      this.speed_up = 0;
    }
    if (this.time > 600) {
      this.time = -this.time;
    }

    // let width = this.canvas.clientWidth;
    // let height = this.canvas.clientHeight
    // for (let x = 10; x < width; x += 5) {
    //   for (let y = 10; y < height; y += 5) {
    //     let n = noise.perlin3(x * 0.005, y * 0.005, Math.abs(this.time) * 0.02);
    //     this.ctx.beginPath();
    //     this.ctx.fillStyle = "rgba(255, 255,255," + Math.abs(n) + ")";
    //     this.ctx.strokeStyle = "rgba(255, 255,255," + Math.abs(n) + ")";

    //     this.ctx.rect(x,y,5, 5);
    //     this.ctx.fill();
    //   }
    // }
  }

  clearCtx() {
    this.ctx.clearRect(0, 0, this.video.videoWidth, this.video.videoHeight);
  }

  dist(x1, y1, x2, y2) {
    var a = x1 - x2;
    var b = y1 - y2;
    return Math.sqrt( a*a + b*b );
  }
  /**
   * Draw the keypoints and skeleton on the video.
   * @param poses A list of poses to render.
   */
  drawResults(poses) {

    for (var i = 0; i < poses.length; i ++) {
      const pose = poses[i];
  
      this.pastPoses.push(pose);

      this.finger.update(pose);

      for (const ball of this.balls) {
        ball.update(this.finger);
      }

      if (this.pastPoses.length == 5) {
        for (const ball of this.balls) {
          ball.pos = this.finger.pos;
        }
      }

      if (this.pastPoses.length > this.maxPoses) {
        this.pastPoses.shift();
      }

      this.drawResult(pose, i);
    }

    for (let i = 0; i < this.pastPoses.length; i++) {
      // this.drawResult(this.pastPoses[i], i);
    }
  }

  /**
   * Draw the keypoints and skeleton on the video.
   * @param pose A pose with keypoints to render.
   * @param index index.
   */
  drawResult(pose, index) {
    if (pose.keypoints != null) {
      this.drawKeypoints(pose.keypoints, index);
      // this.drawSkeleton(pose.keypoints, pose.id, index);
    }
    if (pose.keypoints3D != null && params.STATE.modelConfig.render3D) {
      this.drawKeypoints3D(pose.keypoints3D);
    }

    if (index == 20) {
      // this.drawSkeleton(pose.keypoints, pose.id, this.maxPoses * this.falloffEdge);
    }
    
  }

  dist(x1,y1,x2,y2) {
    return Math.sqrt( Math.pow((x1-x2), 2) + Math.pow((y1-y2), 2) );
  }

  isOpacity(i) {
    return (i == 5 || i == 6 || i == 7 || i == 8 || (i >= 0 && i <= 4));
  }
  /**
   * Draw the keypoints on the video.
   * @param keypoints A list of keypoints.
   */
  drawKeypoints(keypoints, index) {
    // this.drawKeypoint(keypoints[9]);
    const keypointInd =
        posedetection.util.getKeypointIndexBySide(params.STATE.model);


    let width = this.canvas.clientWidth;
    let height = this.canvas.clientHeight
    for (let x = 10; x < width; x += 15) {
      for (let y = 10; y < height; y += 15) {
        let n = noise.perlin3(x * 0.005, y * 0.005, Math.abs(this.time) * 0.02);

        let distance_head = this.dist(keypoints[0].x, keypoints[0].y, x, y);
        let distance_right = this.dist(keypoints[10].x, keypoints[10].y, x, y);
        let distance_left = this.dist(keypoints[9].x, keypoints[9].y, x, y);
        let distance_left_head = this.dist(keypoints[0].x, keypoints[0].y, keypoints[9].x, keypoints[9].y);
        let distance_right_head = this.dist(keypoints[0].x, keypoints[0].y, keypoints[10].x, keypoints[10].y);
        let distance_feet = this.dist(keypoints[15].x, keypoints[15].y, keypoints[16].x, keypoints[16].y);
        let distance_feet_y = this.dist(0, keypoints[15].y, 0, keypoints[16].y);

        if (distance_feet_y > 90) {
          this.speed_up = Math.min(200, distance_feet_y);
        }
        
        // let distance_left_elbow_head = this.dist(keypoints[0].x, keypoints[0].y, keypoints[7].x, keypoints[7].y);
        // let distance_right_elbow_head = this.dist(keypoints[0].x, keypoints[0].y, keypoints[8].x, keypoints[8].y);
        // if (distance_left_elbow_head > 100)  {
        //   distance_left_elbow_head = 100;
        // }
        // if (distance_right_elbow_head > 100)  {
        //   distance_right_elbow_head = 100;
        // }


        let distance = Math.min(Math.min(distance_head, distance_right), distance_left);

        let color_scale = 1.5;
        if (distance_left_head > 255 * color_scale)  {
          distance_left_head = 255 * color_scale;
        }
        if (distance_right_head > 255 * color_scale)  {
          distance_right_head = 255 * color_scale;
        }
        if (distance_feet > 255)  {
          distance_feet = 255;
        }


        n = Math.abs(n);
        n = n / 7;
        n = n / Math.sqrt(distance) * 500 / Math.sqrt(distance);

        this.ctx.beginPath();
        let rgb = hslToRgb(distance_left_head / color_scale / 255 + distance / color_scale / 255, distance_right_head / color_scale / 255 / 2 + distance_left_head / color_scale / 255 / 2, 150 / 255);
        
        
        if (distance_feet > 160) 
        {
          // this.ctx.clearRect(0, 0, this.video.videoWidth, this.video.videoHeight);
          // this.ctx.beginPath();
          // this.ctx.fillStyle = "rgb(200,200,200)";
          // this.ctx.rect(0, 0, this.video.videoWidth, this.video.videoHeight);
          // this.ctx.stroke();
          // this.ctx.fill();
          // this.clearCtx()
          this.to_invert = true;
          
          // this.ctx.beginPath();
          //let rgb = hslToRgb(0.5, distance_right_head / color_scale / 255 / 2 + distance_left_head / color_scale / 255 / 2, 150 / 255);
          this.ctx.fillStyle = "rgba(" + rgb[0] +  "," + rgb[1] + "," + rgb[2] + ","  + n + ")";
          this.ctx.strokeStyle = "rgba(" + rgb[0] +  "," + rgb[1] + "," + rgb[2] + "," + n + ")";
        } 
        else 
        {
          this.ctx.fillStyle = "rgba(" + rgb[0] +  "," + rgb[1] + "," + rgb[2] + ","  + n + ")";
          this.ctx.strokeStyle = "rgba(" + rgb[0] +  "," + rgb[1] + "," + rgb[2] + "," + n + ")";
        }

        //this.ctx.fillStyle = "rgba(" + distance4 / color_scale + "," + gb + "," + gb + ","  + n + ")";
        //this.ctx.strokeStyle = "rgba(100, 100, 100," + n +  ")";

        this.ctx.rect(x,y,15, 15);
        this.ctx.fill();
      }
    }



        // C
    // this.ctx.fillStyle = "rgba(" + this.color + ", " + index / this.maxPoses + ")";
    // this.ctx.strokeStyle = "rgba(" + this.color + ", " + index / this.maxPoses + ")";
    // this.ctx.lineWidth = params.DEFAULT_LINE_WIDTH;

    // var hipR = new Victor(keypoints[12].x, keypoints[12].y);
    // var kneeR = new Victor(keypoints[14].x, keypoints[14].y);
    // var footR = new Victor(keypoints[16].x, keypoints[16].y);
    // var hipL = new Victor(keypoints[11].x, keypoints[11].y);
    // var kneeL = new Victor(keypoints[13].x, keypoints[13].y);
    // var footL = new Victor(keypoints[15].x, keypoints[15].y);

    // var hip_to_knee = kneeR.clone().subtract(hipR);
    // var knee_to_foot = footR.clone().subtract(kneeR);

    // var shoulderR = new Victor(keypoints[6].x, keypoints[6].y);
    // var elbowR = new Victor(keypoints[8].x, keypoints[8].y);
    // var handR = new Victor(keypoints[10].x, keypoints[10].y);
    // var shoulderL = new Victor(keypoints[5].x, keypoints[5].y);
    // var elbowL = new Victor(keypoints[7].x, keypoints[7].y);
    // var handL = new Victor(keypoints[9].x, keypoints[9].y);

    // var shoulder_to_elbow = elbowR.clone().subtract(shoulderR);
    // var elbow_to_hand = handR.clone().subtract(elbowR);

    // var orbR = handR.clone().subtract(elbowR);
    // orbR = handR.clone().add(orbR)
    // var orbL = handL.clone().subtract(elbowL);
    // orbL = handL.clone().add(orbL)

    // this.ctx.fillStyle = "rgba(255,0,0," + index / this.maxPoses / this.falloff  + ")";
    // this.drawBig(orbR.x, orbR.y, keypoints[9].score);

    // for (const ball of this.balls) {
    //   if (this.finger.vel.length() > 15) {
    //     this.ctx.fillStyle = "rgba(255,255,0," + index / this.maxPoses / this.falloff  + ")";
    //   }

    //   else {
    //     this.ctx.fillStyle = "rgba(255,120,0," + index / this.maxPoses / this.falloff  + ")";
    //   }

    //   this.drawBig(ball.pos.x, ball.pos.y, keypoints[10].score);
    // }

    
    // this.ctx.fillStyle = "rgba(" + this.color + ", " + index / this.maxPoses / this.falloff + ")";
    // this.ctx.strokeStyle = "rgba(" + this.color + ", " + index / this.maxPoses / this.falloff + ")";

    // for (const i of keypointInd.middle) {
    //   //this.drawKeypoint(keypoints[i]);
    //   this.drawXY(keypoints[i].x, keypoints[i].y + (this.maxPoses - index ) * this.speed, keypoints[i].score);
    // }


    // for (const i of keypointInd.left) {
    //   if (this.isOpacity(i)) {
    //     this.ctx.fillStyle = "rgba(" + this.color + ", " + index / this.maxPoses / this.falloff + ")";
    //     this.ctx.strokeStyle = "rgba(" + this.color + ", " + index / this.maxPoses / this.falloff + ")";
    //   } else {
    //     this.ctx.fillStyle = "rgba(" + this.color + ", " + index / this.maxPoses + ")";
    //     this.ctx.strokeStyle = "rgba(" + this.color + ", " + index / this.maxPoses + ")";
    //   }
    //   this.drawXY(keypoints[i].x, keypoints[i].y + (this.maxPoses - index ) * this.speed, keypoints[i].score);
    // }

    // for (const i of keypointInd.right) {
    //   if (this.isOpacity(i)) {
    //     this.ctx.fillStyle = "rgba(" + this.color + ", " + index / this.maxPoses / this.falloff + ")";
    //     this.ctx.strokeStyle = "rgba(" + this.color + ", " + index / this.maxPoses / this.falloff + ")";
    //   } else {
    //     this.ctx.fillStyle = "rgba(" + this.color + ", " + index / this.maxPoses + ")";
    //     this.ctx.strokeStyle = "rgba(" + this.color + ", " + index / this.maxPoses + ")";
    //   }
    //   this.drawXY(keypoints[i].x, keypoints[i].y + (this.maxPoses - index ) * this.speed, keypoints[i].score);
    
    // }

    // let x_tmp;
    // let y_tmp;

    // this.ctx.fillStyle = "rgba(" + this.color + ", " + index / this.maxPoses + ")";
    // this.ctx.strokeStyle = "rgba(" + this.color + ", " + index / this.maxPoses + ")";

    // x_tmp = (keypoints[8].x + keypoints[10].x) / 2;
    // y_tmp = (keypoints[8].y + keypoints[10].y) / 2;
    // this.drawXY(x_tmp, y_tmp + (this.maxPoses - index ) * this.speed, keypoints[8].score);

    // x_tmp = (keypoints[7].x + keypoints[9].x) / 2;
    // y_tmp = (keypoints[7].y + keypoints[9].y) / 2;
    // this.drawXY(x_tmp, y_tmp + (this.maxPoses - index ) * this.speed, keypoints[7].score);

    // this.ctx.fillStyle = "rgba(" + this.color + ", " + index / this.maxPoses / this.falloff + ")";
    // this.ctx.strokeStyle = "rgba(" + this.color + ", " + index / this.maxPoses / this.falloff + ")";

    // x_tmp = (keypoints[8].x + keypoints[6].x) / 2;
    // y_tmp = (keypoints[8].y + keypoints[6].y) / 2;
    // this.drawXY(x_tmp, y_tmp + (this.maxPoses - index ) * this.speed, keypoints[6].score);

    // x_tmp = (keypoints[5].x + keypoints[7].x) / 2;
    // y_tmp = (keypoints[5].y + keypoints[7].y) / 2;
    // this.drawXY(x_tmp, y_tmp + (this.maxPoses - index ) * this.speed, keypoints[5].score);
  
  }

  drawXY(x, y, s) {
    const score = s != null ? s : 1;
    const scoreThreshold = params.STATE.modelConfig.scoreThreshold || 0;

    if (score >= scoreThreshold) {
      const circle = new Path2D();
      circle.arc(x, y, params.DEFAULT_RADIUS, 0, 2 * Math.PI);
      this.ctx.fill(circle);
      this.ctx.stroke(circle);
    }
  }

  drawBig(x, y, s) {
    const score = s != null ? s : 1;
    const scoreThreshold = params.STATE.modelConfig.scoreThreshold || 0;

    if (score >= scoreThreshold) {
      const circle = new Path2D();
      circle.arc(x, y, 40, 0, 2 * Math.PI);
      this.ctx.fill(circle);
      this.ctx.stroke(circle);
    }
  }

  drawKeypoint(keypoint) {
    // If score is null, just show the keypoint.
    const score = keypoint.score != null ? keypoint.score : 1;
    const scoreThreshold = params.STATE.modelConfig.scoreThreshold || 0;

    if (score >= scoreThreshold) {
      const circle = new Path2D();
      circle.arc(keypoint.x, keypoint.y, params.DEFAULT_RADIUS, 0, 2 * Math.PI);
      this.ctx.fill(circle);
      this.ctx.stroke(circle);
    }
  }

  drawSkeletonLine(point1, point2, index) {
    this.ctx.fillStyle = "rgba(" + this.color2 + ", " + index / this.maxPoses / this.falloffEdge + ")";
    this.ctx.strokeStyle = "rgba(" + this.color2 + ", " + index / this.maxPoses / this.falloffEdge + ")";

    this.ctx.beginPath();
    this.ctx.moveTo(point1.x, point1.y);
    this.ctx.lineTo(point2.x, point2.y);
    this.ctx.stroke();
  }
  /**
   * Draw the skeleton of a body on the video.
   * @param keypoints A list of keypoints.
   */
  drawSkeleton(keypoints, poseId, index) {
    // Each poseId is mapped to a color in the color palette.
    const color = params.STATE.modelConfig.enableTracking && poseId != null ?
        COLOR_PALETTE[poseId % 20] :
        'White';
    this.ctx.fillStyle = "rgba(" + this.color2 + ", " + index / this.maxPoses / this.falloffEdge + ")";
    this.ctx.strokeStyle = "rgba(" + this.color2 + ", " + index / this.maxPoses / this.falloffEdge + ")";
    // this.ctx.fillStyle = "White";
    // this.ctx.strokeStyle = "White";
    this.ctx.lineWidth = params.DEFAULT_LINE_WIDTH;

    posedetection.util.getAdjacentPairs(params.STATE.model).forEach(([
                                                                      i, j
                                                                    ]) => {
      if (i >= 0 && i <= 4) {
        return;
      }
      const kp1 = keypoints[i];
      const kp2 = keypoints[j];

      // If score is null, just show the keypoint.
      const score1 = kp1.score != null ? kp1.score : 1;
      const score2 = kp2.score != null ? kp2.score : 1;
      const scoreThreshold = params.STATE.modelConfig.scoreThreshold || 0;

      if (score1 >= scoreThreshold && score2 >= scoreThreshold) {
        this.ctx.beginPath();
        this.ctx.moveTo(kp1.x, kp1.y);
        this.ctx.lineTo(kp2.x, kp2.y);
        this.ctx.stroke();
      }
    });
  }

  drawKeypoints3D(keypoints) {
    const scoreThreshold = params.STATE.modelConfig.scoreThreshold || 0;
    const pointsData =
        keypoints.map(keypovar=> ([-keypoint.x, -keypoint.y, -keypoint.z]));

    const dataset =
        new scatter.ScatterGL.Dataset([...pointsData, ...ANCHOR_POINTS]);

    const keypointInd =
        posedetection.util.getKeypointIndexBySide(params.STATE.model);
    this.scatterGL.setPointColorer((i) => {
      if (keypoints[i] == null || keypoints[i].score < scoreThreshold) {
        // hide anchor points and low-confident points.
        return '#ffffff';
      }
      if (i === 0) {
        return '#ff0000' /* Red */;
      }
      if (keypointInd.left.indexOf(i) > -1) {
        return '#00ff00' /* Green */;
      }
      if (keypointInd.right.indexOf(i) > -1) {
        return '#ffa500' /* Orange */;
      }
    });

    if (!this.scatterGLHasInitialized) {
      this.scatterGL.render(dataset);
    } else {
      this.scatterGL.updateDataset(dataset);
    }
    const connections = posedetection.util.getAdjacentPairs(params.STATE.model);
    const sequences = connections.map(pair => ({indices: pair}));
    this.scatterGL.setSequences(sequences);
    this.scatterGLHasInitialized = true;
  }
}
