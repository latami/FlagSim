/* Flagsim version 0.11. Simple flag simulator simulating flag with free-moving pole
 * Copyright (C) 2014 Taneli Mikkonen
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.*/


/* Compiling:
 * gcc -O3 -Wall -lm -lSDL -lGL -pthread Flagsim.c -o Flagsim
 *
 * glibc version older than 2.17 may require -lrt be added to compiling command.
 * You can check version by issuing command 'ldd --version' or '/lib/libc.so.6'
 * in terminal.
 *
 *
 * default flag assembly moment of inertia assumes 1.2192 m (4 feet) horizontal
 * tube with diameter of 0.0254 m (1 inch). Tube wall is assumed to be
 * 7.94*10⁻⁴ m (1/32 inches) and made of aluminium (2700 kg/m³). Vertical part
 * of the flag assembly has negligible effect on moment of inertia.
 *
 * Flag dimensions are 1.2192 m x 0.762 m (4 by 2.5 feet). Mass of the flag is
 * guesstimated by assuming 125 g/m².
 *
 * Spring constants are pretty much random. Doesn't try to accurately simulate
 * any material because this simple model is not meant for it.
 *
 * http://www.jsc.nasa.gov/history/flag/flag.htm (Construction and Testing)
 * http://history.nasa.gov/alsj/a15/A15_PressKit.pdf
 * (APOLLO 15 FLAGS, LUNAR MODULE PLAQUE) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include <GL/gl.h>
#include <SDL/SDL.h>

#include <pthread.h>

#define HNODES 32       /* Flag horizontal nodes */
#define VNODES 20       /* Flag vertical nodes */
#define SPRINGL 0.0381  /* Spring length between nodes */

#define SCREENW 800
#define SCREENH 600

typedef struct Node Node;

struct Node {
    float pos[3];
    float vel[3];
    float forces1[3];
    float forces2[3];
    float invMass;
    Node *s[4];
};

struct gfThreadData {
    Node *flag;
    Node *assembly;
    int exitFlag;
    int viewFlag;
    double *simulationTime;

    pthread_mutex_t *lockPtr;
} gfThreadData;

struct integratorData {
    Node *flag;
    Node *flagTmp;
    int fNodes;             /* Amount of flag nodes */
    Node *assembly;
    Node *assemblyTmp;
    int aNodes;             /* Amount of flag assembly nodes */
    float J;               /* Moment of inertia for the flag assembly */
    float alfaPos;         /* angular position */
    float alfaVel;         /* angular velocity */
    float alfa1;           /* angular acceleration */
    float alfa2;
    float Rot_fric;        /* rotational friction */
    float appliedForceZ;
    float k_stretch;
    float k_pend;
    float c;
    float gee;             /* Gravitational acceleration */
    int leapFrogFirstIteration;

    pthread_mutex_t lockData;
};


void keyPresses(SDL_Event *event, struct gfThreadData *thData) {

        switch (event->key.keysym.sym) {
            case SDLK_ESCAPE:
                thData->exitFlag = 1;
                break;
            case SDLK_t:
                printf("Elapsed time: %.2lf s\n", (*thData->simulationTime));
                break;
            case SDLK_F1:
                thData->viewFlag = 0;
                break;
            case SDLK_F2:
                thData->viewFlag = 1;
                break;
            default:
                break;
        }
}

void eventHandling(SDL_Event *event, struct gfThreadData *thData) {

    while ( SDL_PollEvent( event ) ) {
        switch( event->type ) {
            case SDL_QUIT:
                thData->exitFlag = 1;
                break;
            case SDL_KEYDOWN:
                keyPresses(event, thData);
                break;
            default:
                break;
        }
    }
}

int initVideo(SDL_Surface **surface, SDL_VideoInfo const **videoInfo) {
    double fovRadian = 0.7854, drawNear = 0.1, drawFar = 100.0, drawTop, drawRight;

    GLfloat sun_emission[] = { 1.0, 1.0, 1.0, 1 };
    GLfloat ambientlight[4]={0.1, 0.1, 0.1, 1};

    if ( SDL_Init(SDL_INIT_VIDEO) < 0 ) {
        fprintf(stderr, "Video init failed: %s\n", SDL_GetError());
        SDL_Quit();
        return EXIT_FAILURE;
    }

    (*videoInfo) = SDL_GetVideoInfo();
    if ( !(*videoInfo) ) {
        fprintf(stderr, "Video query failed: %s\n", SDL_GetError());
        SDL_Quit();
        return EXIT_FAILURE;
    }

    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    (*surface) = SDL_SetVideoMode(SCREENW, SCREENH, 0, SDL_OPENGL);

    GLint dBuffer;
    SDL_GL_GetAttribute(SDL_GL_DOUBLEBUFFER, &dBuffer);
    if (!dBuffer)
        fprintf(stderr, "Couldn't set doublebuffering!\n");

    if ( !(*surface) ) {
        fprintf(stderr, "Video mode set failed: %s\n", SDL_GetError());
        SDL_Quit();
        return EXIT_FAILURE;
    }

    glShadeModel(GL_SMOOTH);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, sun_emission);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambientlight);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);

    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    glViewport(0, 0, SCREENW, SCREENH);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    drawTop = tan(fovRadian*0.5) * drawNear;
    drawRight = ((double)SCREENW/(double)SCREENH) * drawTop;
    glFrustum(-drawRight, drawRight, -drawTop, drawTop, drawNear, drawFar);
    glMatrixMode(GL_MODELVIEW);

    SDL_WM_SetCaption("Flag simulation", "");

    return EXIT_SUCCESS;
}

void calcNormal(float *p1, float *p2, float *p3, float *result) {
    float l1[3], l2[3], l;

    l1[0]=p2[0]-p1[0];
    l1[1]=p2[1]-p1[1];
    l1[2]=p2[2]-p1[2];

    l2[0]=p3[0]-p1[0];
    l2[1]=p3[1]-p1[1];
    l2[2]=p3[2]-p1[2];

    result[0] = l1[1]*l2[2] - l1[2]*l2[1];
    result[1] = l1[2]*l2[0] - l1[0]*l2[2];
    result[2] = l1[0]*l2[1] - l1[1]*l2[0];

    l = sqrtf(result[0]*result[0] + result[1]*result[1] + result[2]*result[2]);
    result[0] /= l;
    result[1] /= l;
    result[2] /= l;
}

void drawFlag(Node *flag) {
    int i, red = 0;
    float n[3];

    for (i=0; i < VNODES*HNODES; i++) {
        if ((flag[i].s[1] != NULL) && (flag[i].s[2] != NULL)) {
            if ((flag[i+1].s[2] != NULL) && (flag[i].s[2]->s[1] != NULL)) {
                if (i % HNODES == 0) {
                    if (red == 0) {
                        glColor3f(1.0f, 0.0f, 0.0f);
                        red = 1;
                    }
                    else {
                        if (red == 1) {
                        glColor3f(1.0f, 1.0f, 1.0f);
                        red = 0;
                        }
                    }
                }

                glBegin(GL_QUADS);
                calcNormal(flag[i].pos, flag[i].s[1]->pos, flag[i].s[2]->pos, n);
                glNormal3fv(n);
                glVertex3fv(flag[i].pos);

                calcNormal(flag[i].s[2]->pos, flag[i].pos, flag[i].s[2]->s[1]->pos, n);
                glNormal3fv(n);
                glVertex3fv(flag[i].s[2]->pos);

                calcNormal(flag[i].s[2]->s[1]->pos, flag[i].s[2]->pos, flag[i].s[1]->pos, n);
                glNormal3fv(n);
                glVertex3fv(flag[i].s[2]->s[1]->pos);

                calcNormal(flag[i].s[1]->pos, flag[i].s[2]->s[1]->pos, flag[i].pos, n);
                glNormal3fv(n);
                glVertex3fv(flag[i].s[1]->pos);
                glEnd();
            }
        }
    }
}

void drawPole(Node *pole) {
    glDisable(GL_LIGHTING);
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3fv(pole[0].pos);
    glVertex3fv(pole[HNODES-1].pos);
    glEnd();
    glBegin(GL_LINES);
    glVertex3fv(pole[HNODES].pos);
    glVertex3fv(pole[HNODES+VNODES-1].pos);
    glEnd();
    glEnable(GL_LIGHTING);
}

void drawGraphics(void *arguments) {
    struct gfThreadData *threadData;

    const SDL_VideoInfo *videoInfo;
    SDL_Surface *surface;
    SDL_Event event;

    threadData = (struct gfThreadData *) arguments;

    /* Test that we get graphics */
    if ( initVideo(&surface, &videoInfo) != EXIT_SUCCESS ) {
        threadData->exitFlag = 1;
        /* Do we need SDL_Quit() if video init fails? */
        return;
    }

    float lightPos[]={-0.2f, -0.3f, -1.0f, 0.0f};

    while (threadData->exitFlag == 0) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();

        glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

        /* 2 viewing directions */
        if (threadData->viewFlag == 0) {
            glTranslatef(-0.7f, 0.4f, -2.0f);
            glRotatef(-50.0f, 0.0f, 1.0f, 0.0f);
        }
        else {
            glTranslatef(-0.3f, 0.0f, -2.0f);
            glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
        }

        /* Locking data to make sure it doesn't change while drawing
         * Not really necessary for this application, but for the sake of completeness */
        pthread_mutex_lock(threadData->lockPtr);
        drawFlag(threadData->flag);
        drawPole(threadData->assembly);
        pthread_mutex_unlock(threadData->lockPtr);

        SDL_GL_SwapBuffers();
        eventHandling(&event, threadData);
        SDL_Delay(16); /* Just a dumb wait */
    }
    SDL_Quit();
}

void springForce(float *pos1, float *pos2, float *vel1, float *vel2
                 , float *res, float k, float c) {
    float r, f, f1, f2, f3, x;

    r = sqrtf( (pos2[0]-pos1[0])*(pos2[0]-pos1[0])
            + (pos2[1]-pos1[1])*(pos2[1]-pos1[1])
            + (pos2[2]-pos1[2])*(pos2[2]-pos1[2]) );

     f = k/r;
     f1 = f*(pos2[0]-pos1[0]);
     f2 = f*(pos2[1]-pos1[1]);
     f3 = f*(pos2[2]-pos1[2]);

     /* Displacement */
     x = r-SPRINGL;

     res[0] += f1*x + c*(vel2[0]-vel1[0]);
     res[1] += f2*x + c*(vel2[1]-vel1[1]);
     res[2] += f3*x + c*(vel2[2]-vel1[2]);
}

/* Quick and dirty pending resistance. Using faces would have been more correct
 * than using parallel lines */
void pendingForce(float *pos1, float *pos2, float *posCenterNode, float *vel1, float *vel2, float *velCenterNode, float *res, float k, float c) {
    float posTmp[3], velTmp[3];

    /* Midpoint of two position vectors */
    posTmp[0] = (pos2[0]+pos1[0])*0.5;
    posTmp[1] = (pos2[1]+pos1[1])*0.5;
    posTmp[2] = (pos2[2]+pos1[2])*0.5;

    velTmp[0] = (vel2[0]+vel1[0])*0.5;
    velTmp[1] = (vel2[1]+vel1[1])*0.5;
    velTmp[2] = (vel2[2]+vel1[2])*0.5;

    /* Assume midpoint is equilibrium position */
    res[0] = k*(posTmp[0]-posCenterNode[0]) + c*(velTmp[0]-velCenterNode[0]);
    res[1] = k*(posTmp[1]-posCenterNode[1]) + c*(velTmp[1]-velCenterNode[1]);
    res[2] = k*(posTmp[2]-posCenterNode[2]) + c*(velTmp[2]-velCenterNode[2]);
}

float friction(float vel, float friction_acc, float acc, float threshold, float dt) {
    float fric = 0.0;
    int sign1, sign2;


    fric = -copysignf(friction_acc, vel);

    /* friction starts to increase then going under threshold */
    if (fabsf(vel < threshold)) {
        fric += fric * 0.25f*(threshold-vel)/threshold;
    }

    /* Check if rotation is going to change direction */
    sign1 = signbit( vel + 0.5*(acc+fric)*dt );
    sign2 = signbit( vel );

    if ( (sign1 == 0 && sign2 != 0) || (sign1 != 0 && sign2 == 0) ) {
        /* Limit friction to avoid acceleration */
        fric = -2.0f*vel/dt + acc;
    }

    return fric;
}

/* leapfrog / velocity verlet */
void intLeapfrog(struct integratorData *vals, float dt) {
    int i, j;
    float pForce[3], M, kPend, kStretch, fric;
    Node *p, *pTmp;

    kStretch = vals->k_stretch;
    kPend = vals->k_pend;

    /* Skip first acceleration calculations that depend position
     * as they are already calculated in the second stage of
     * previous iteration */
    if (vals->leapFrogFirstIteration == 1) {
        for(i=0; i < vals->fNodes; i++) {
            vals->flag[i].forces1[0] = 0.0f;
            vals->flag[i].forces1[1] = 0.0f;
            vals->flag[i].forces1[2] = 0.0f;
            vals->flag[i].forces2[0] = 0.0f;
            vals->flag[i].forces2[1] = 0.0f;
            vals->flag[i].forces2[2] = 0.0f;
        }
        for(i=0; i < vals->aNodes; i++) {
            vals->assembly[i].forces1[0] = 0.0f;
            vals->assembly[i].forces1[1] = 0.0f;
            vals->assembly[i].forces1[2] = 0.0f;
            vals->assembly[i].forces2[0] = 0.0f;
            vals->assembly[i].forces2[1] = 0.0f;
            vals->assembly[i].forces2[2] = 0.0f;
        }

        /* Half of the force in opposite direction */
        vals->flag[vals->fNodes-1].forces1[2] = vals->appliedForceZ*0.5;
        vals->assembly[HNODES-1].forces1[2] -= vals->appliedForceZ*0.5;

        for (i=0; i < vals->fNodes; i++) {
            /* Avoid some unnecessary pointer passage */
            p = &vals->flag[i];

            for (j=0; j < 4; j++) {
                if (p->s[j] != NULL)
                    springForce( p->pos, p->s[j]->pos, p->vel, p->s[j]->vel, p->forces1, kStretch, vals->c);
            }
            if ( (p->s[0] != NULL) && (p->s[2] != NULL) ) {

                pendingForce(p->s[2]->pos, p->s[0]->pos, p->pos, p->s[2]->vel, p->s[0]->vel, p->vel, pForce, kPend, vals->c);
                p->forces1[0] += pForce[0];
                p->forces1[1] += pForce[1];
                p->forces1[2] += pForce[2];
                /* Newtons 3rd law */
                p->s[0]->forces1[0] -= pForce[0]*0.5;
                p->s[0]->forces1[1] -= pForce[1]*0.5;
                p->s[0]->forces1[2] -= pForce[2]*0.5;
                p->s[2]->forces1[0] -= pForce[0]*0.5;
                p->s[2]->forces1[1] -= pForce[1]*0.5;
                p->s[2]->forces1[2] -= pForce[2]*0.5;
            }
            if ( (p->s[3] != NULL) && (p->s[1] != NULL) ) {

                pendingForce(p->s[3]->pos, p->s[1]->pos, p->pos, p->s[3]->vel, p->s[1]->vel, p->vel, pForce, kPend, vals->c);
                p->forces1[0] += pForce[0];
                p->forces1[1] += pForce[1];
                p->forces1[2] += pForce[2];
                p->s[3]->forces1[0] -= pForce[0]*0.5;
                p->s[3]->forces1[1] -= pForce[1]*0.5;
                p->s[3]->forces1[2] -= pForce[2]*0.5;
                p->s[1]->forces1[0] -= pForce[0]*0.5;
                p->s[1]->forces1[1] -= pForce[1]*0.5;
                p->s[1]->forces1[2] -= pForce[2]*0.5;
            }

            /* Gravitational force */
            p->forces1[1] += -vals->gee/p->invMass;
        }

        /* Angular acceleration */
        M = 0.0;
        for (i=0; i < HNODES; i++) {
            p = &vals->assembly[i];

            springForce( p->pos, p->s[2]->pos, p->vel, p->s[2]->vel, p->forces1, kStretch, vals->c);

            M += p->pos[0]*p->forces1[2] - p->pos[2]*p->forces1[0];
        }
        vals->alfa1 = M/vals->J;

    }
    else { /* This executes always after first iteration (if not disabled) */
        for(i=0; i < vals->fNodes; i++) {
            vals->flag[i].forces1[0] = vals->flag[i].forces2[0];
            vals->flag[i].forces1[1] = vals->flag[i].forces2[1];
            vals->flag[i].forces1[2] = vals->flag[i].forces2[2];
            vals->flag[i].forces2[0] = 0.0f;
            vals->flag[i].forces2[1] = 0.0f;
            vals->flag[i].forces2[2] = 0.0f;
        }
        for(i=0; i < vals->aNodes; i++) {
            vals->assembly[i].forces1[0] = vals->assembly[i].forces2[0];
            vals->assembly[i].forces1[1] = vals->assembly[i].forces2[1];
            vals->assembly[i].forces1[2] = vals->assembly[i].forces2[2];
            vals->assembly[i].forces2[0] = 0.0f;
            vals->assembly[i].forces2[1] = 0.0f;
            vals->assembly[i].forces2[2] = 0.0f;
        }
        vals->alfa1 = vals->alfa2;
    }

    fric = friction(vals->alfaVel, vals->Rot_fric/vals->J, vals->alfa1, 0.001745f, dt);

    /* New angular position */
    vals->alfaPos = vals->alfaPos + vals->alfaVel*dt + 0.5*(vals->alfa1+fric)*dt*dt;

    /* New positions for nodes in horizontal part of a assembly */
    for (i=0; i < HNODES; i++) {
        vals->assemblyTmp[i].pos[0] = (SPRINGL+SPRINGL*i)*cosf(vals->alfaPos);
        vals->assemblyTmp[i].pos[2] = (SPRINGL+SPRINGL*i)*sinf(vals->alfaPos);
    }

    /* New positions */
    for (i=0; i < vals->fNodes; i++) {
        p = &vals->flag[i];
        pTmp = &vals->flagTmp[i];
        pTmp->pos[0] = p->pos[0]+p->vel[0]*dt+0.5f*p->forces1[0]*p->invMass*dt*dt;
        pTmp->pos[1] = p->pos[1]+p->vel[1]*dt+0.5f*p->forces1[1]*p->invMass*dt*dt;
        pTmp->pos[2] = p->pos[2]+p->vel[2]*dt+0.5f*p->forces1[2]*p->invMass*dt*dt;
        /* Approximate velocities for damping purposes */
        pTmp->vel[0] = p->vel[0]+p->forces1[0]*p->invMass*dt;
        pTmp->vel[1] = p->vel[1]+p->forces1[1]*p->invMass*dt;
        pTmp->vel[2] = p->vel[2]+p->forces1[2]*p->invMass*dt;
    }


    /* ################### Second stage ########################### */

    vals->flag[vals->fNodes-1].forces2[2] = vals->appliedForceZ*0.5;
    vals->assembly[HNODES-1].forces2[2] -= vals->appliedForceZ*0.5;

    for(i=0; i < vals->fNodes; i++) {
        p = &vals->flag[i];
        pTmp = &vals->flagTmp[i];

        for (j=0; j < 4; j++) {
            if (pTmp->s[j] != NULL)
                springForce( pTmp->pos, pTmp->s[j]->pos, pTmp->vel, pTmp->s[j]->vel, p->forces2, kStretch, vals->c);
        }
        if ( (p->s[0] != NULL) && (p->s[2] != NULL) ) {

            pendingForce(pTmp->s[2]->pos, pTmp->s[0]->pos, pTmp->pos, pTmp->s[2]->vel, pTmp->s[0]->vel, pTmp->vel, pForce, kPend, vals->c);
            p->forces2[0] += pForce[0];
            p->forces2[1] += pForce[1];
            p->forces2[2] += pForce[2];
            p->s[0]->forces2[0] -= pForce[0]*0.5;
            p->s[0]->forces2[1] -= pForce[1]*0.5;
            p->s[0]->forces2[2] -= pForce[2]*0.5;
            p->s[2]->forces2[0] -= pForce[0]*0.5;
            p->s[2]->forces2[1] -= pForce[1]*0.5;
            p->s[2]->forces2[2] -= pForce[2]*0.5;
        }
        if ( (p->s[3] != NULL) && (p->s[1] != NULL) ) {

            pendingForce(pTmp->s[3]->pos, pTmp->s[1]->pos, pTmp->pos, pTmp->s[3]->vel, pTmp->s[1]->vel, pTmp->vel, pForce, kPend, vals->c);
            p->forces2[0] += pForce[0];
            p->forces2[1] += pForce[1];
            p->forces2[2] += pForce[2];
            p->s[3]->forces2[0] -= pForce[0]*0.5;
            p->s[3]->forces2[1] -= pForce[1]*0.5;
            p->s[3]->forces2[2] -= pForce[2]*0.5;
            p->s[1]->forces2[0] -= pForce[0]*0.5;
            p->s[1]->forces2[1] -= pForce[1]*0.5;
            p->s[1]->forces2[2] -= pForce[2]*0.5;
        }

        /* Gravitational force */
        p->forces2[1] += -vals->gee/p->invMass;
    }

    /* Angular acceleration */
    M = 0.0;
    for (i=0; i < HNODES; i++) {
        p = &vals->assembly[i];
        pTmp = &vals->assemblyTmp[i];

        springForce( pTmp->pos, pTmp->s[2]->pos, pTmp->vel, pTmp->s[2]->vel, p->forces2, kStretch, vals->c);

        M += pTmp->pos[0]*p->forces2[2] - pTmp->pos[2]*p->forces2[0];
    }
    vals->alfa2 = M/vals->J;

    vals->alfaVel = vals->alfaVel + 0.5*(vals->alfa1+vals->alfa2+fric*2.0)*dt;

    /* Velocity vector for assembly nodes is not needed so we don't calculate them */

    /* New velocities */
    for (i=0; i < vals->fNodes; i++) {
        p = &vals->flag[i];
        pTmp = &vals->flagTmp[i];

        pTmp->vel[0] = p->vel[0]+0.5f*(p->forces1[0]+p->forces2[0])*p->invMass*dt;
        pTmp->vel[1] = p->vel[1]+0.5f*(p->forces1[1]+p->forces2[1])*p->invMass*dt;
        pTmp->vel[2] = p->vel[2]+0.5f*(p->forces1[2]+p->forces2[2])*p->invMass*dt;
    }

    /* Lock mutex to write new values */
    pthread_mutex_lock(&vals->lockData);

    for(i=0; i < vals->fNodes; i++) {
        p = &vals->flag[i];
        pTmp = &vals->flagTmp[i];

        p->vel[0] = pTmp->vel[0];
        p->vel[1] = pTmp->vel[1];
        p->vel[2] = pTmp->vel[2];
        p->pos[0] = pTmp->pos[0];
        p->pos[1] = pTmp->pos[1];
        p->pos[2] = pTmp->pos[2];
    }
    for (i=0; i < HNODES; i++) {
        vals->assembly[i].pos[0] = vals->assemblyTmp[i].pos[0];
        vals->assembly[i].pos[2] = vals->assemblyTmp[i].pos[2];
    }
    /* Release mutex */
    pthread_mutex_unlock(&vals->lockData);

    /* First iteration done */
    vals->leapFrogFirstIteration = 0;
}

/* Returns pointer for a nodearray sizeof V*H */
int createRectangle(Node **rect, int V, int H, float mass) {
    int i, j;

    (*rect) = calloc(sizeof(Node), V*H);
    if ((*rect) == NULL) {
        fprintf(stderr, "Allocation failed in createRectangle()\n");
        return EXIT_FAILURE;
    }

    for (i=0; i < V; i++) {
        for (j=0; j < H; j++) {

            (*rect)[i*H+j].invMass = (V*H)/mass;

            (*rect)[i*H+j].pos[0] = (float)(j+1)*SPRINGL;
            (*rect)[i*H+j].pos[1] = (float)(i+1)*(-SPRINGL);
            (*rect)[i*H+j].pos[2] = 0.0f;

            (*rect)[i*H+j].vel[0] = 0.0f;
            (*rect)[i*H+j].vel[1] = 0.0f;
            (*rect)[i*H+j].vel[2] = 0.0f;

            (*rect)[i*H+j].s[0] = NULL;
            (*rect)[i*H+j].s[1] = NULL;
            (*rect)[i*H+j].s[2] = NULL;
            (*rect)[i*H+j].s[3] = NULL;

            if ( i > 0 )
                (*rect)[i*H+j].s[0] = &(*rect)[(i-1)*H+j];
            if ( j < (H-1) )
                (*rect)[i*H+j].s[1] = &(*rect)[i*H+j+1];
            if ( i < (V-1) )
                (*rect)[i*H+j].s[2] = &(*rect)[(i+1)*H+j];
            if ( j > 0 )
                (*rect)[i*H+j].s[3] = &(*rect)[i*H+j-1];
        }
    }

    return EXIT_SUCCESS;
}

/* Returns pointer for a nodearray sizeof V+H */
int createAssembly(Node **assembly, int V, int H) {
    int i;

    (*assembly) = calloc(sizeof(Node), V+H);
    if ((*assembly) == NULL) {
        fprintf(stderr, "Allocation failed in createAssembly()\n");
        return EXIT_FAILURE;
    }

    /* No need to set node masses */

    for (i=0; i < H; i++) {
        (*assembly)[i].pos[0] = (float)i*SPRINGL+SPRINGL;
        (*assembly)[i].pos[1] = 0.0f;
        (*assembly)[i].pos[2] = 0.0f;
        (*assembly)[i].vel[0] = 0.0f;
        (*assembly)[i].vel[1] = 0.0f;
        (*assembly)[i].vel[2] = 0.0f;
        (*assembly)[i].s[0] = NULL;
        (*assembly)[i].s[1] = NULL;
        (*assembly)[i].s[2] = NULL;
        (*assembly)[i].s[3] = NULL;

    }
    for (i=0; i < V; i++) {
        (*assembly)[H+i].pos[0] = 0.0f;
        (*assembly)[H+i].pos[1] = (float)(i+1)*(-SPRINGL);
        (*assembly)[H+i].pos[2] = 0.0f;
        (*assembly)[H+i].vel[0] = 0.0f;
        (*assembly)[H+i].vel[1] = 0.0f;
        (*assembly)[H+i].vel[2] = 0.0f;
    }

    return EXIT_SUCCESS;
}

/* Assumes flag geometry */
float calcEnergy(Node *p, int n, float kstretch, float kpend, float g0) {
    int i, j, edges = 0;
    float E_k = 0.0, E_p = 0.0, X[3];

    for (i = 0; i < n; i++) {

        E_k += 0.5*(p[i].vel[0]*p[i].vel[0] + p[i].vel[1]*p[i].vel[1]
                  + p[i].vel[2]*p[i].vel[2])/p[i].invMass;

        for (j = 0; j < 4; j++) {
            /* Assume that s[3] spring is already calculated except for the
             * first in a row */
            if ((i % HNODES != 0) && (j==3))
                break;
            /* Assume s[0] is calculated after first row */
            if ((i >= HNODES) && (j==0)) {
                j=1;
            }
            if (p[i].s[j] != NULL) {
                E_p += fabs( 0.5*kstretch*pow( (sqrt( (p[i].pos[0]-p[i].s[j]->pos[0])
                                               *(p[i].pos[0]-p[i].s[j]->pos[0])
                                              + (p[i].pos[1]-p[i].s[j]->pos[1])
                                               *(p[i].pos[1]-p[i].s[j]->pos[1])
                                              + (p[i].pos[2]-p[i].s[j]->pos[2])
                                               *(p[i].pos[2]-p[i].s[j]->pos[2]) ) - SPRINGL ), 2) );
                edges++;
            }
        }
        /* Pending potentials */
        if ( (p[i].s[0] != NULL) && (p[i].s[2] != NULL) ) {
            X[0] = (p[i].s[0]->pos[0]+p[i].s[2]->pos[0])*0.5 - p[i].pos[0];
            X[1] = (p[i].s[0]->pos[1]+p[i].s[2]->pos[1])*0.5 - p[i].pos[1];
            X[2] = (p[i].s[0]->pos[2]+p[i].s[2]->pos[2])*0.5 - p[i].pos[2];
            E_p += 0.5*kpend*(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]);
        }
        if ( (p[i].s[1] != NULL) && (p[i].s[3] != NULL) ) {
            X[0] = (p[i].s[1]->pos[0]+p[i].s[3]->pos[0])*0.5 - p[i].pos[0];
            X[1] = (p[i].s[1]->pos[1]+p[i].s[3]->pos[1])*0.5 - p[i].pos[1];
            X[2] = (p[i].s[1]->pos[2]+p[i].s[3]->pos[2])*0.5 - p[i].pos[2];
            E_p += 0.5*kpend*(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]);
        }

        /* Zero gravitational potential is set below lowest flag node */
        E_p += g0*(p[i].pos[1] - (-2.0))/p[i].invMass;
    }

    printf("Calculated tension potential of %d edges for %d nodes\n", edges, n);
    return E_k + E_p;
}

float calcAngularEnergy(float J, float angularVelocity) {

    return 0.5*J*angularVelocity*angularVelocity;
}

/* float conversion with error checking */
float parse_float( const char *str, int *d_error ) {

    char *pend;
    float f;

    f = strtof(str, &pend);
    if (pend == str || (*pend) != '\0')
        (*d_error) = 1;
    else
        (*d_error) = 0;

    return f;
}

int main(int argc, char **argv)
{

    struct gfThreadData threadData;
    pthread_t renderThread;

    struct integratorData intData;

    float temp, dt = 0.0001, forceTime = 1.0, massFlag = 0.116;
    double runtime = 25.0, timeElapsed = 0.0;
    int i, retVal = 0, error, negativeVal = 0;
    char c;

    /* Some default values*/

    intData.J = 0.1001;             /* Moment of inertia */
    intData.k_stretch = 400.0;
    intData.k_pend = 3.0;
    intData.appliedForceZ = 0.01;
    intData.gee = 1.622;
    intData.Rot_fric = 0.005;
    intData.c = 0.0001;

    intData.leapFrogFirstIteration = 1;

    /* Command line option parsing */
    while (1) {
        c = getopt(argc, argv, "t:T:f:k:K:J:g:r:m:p:c:");
        if (c == -1)
            break;

        switch (c) {
            case 't':
                dt = parse_float(optarg, &error);
                if (error == 1) {
                    fprintf(stderr, "Error parsing stepsize.\n");
                    return EXIT_FAILURE;
                }
                if (dt < 0) negativeVal = 1;
                break;
            case 'T':
                forceTime = parse_float(optarg, &error);
                if (error == 1) {
                    fprintf(stderr, "Error parsing force-applying time.\n");
                    return EXIT_FAILURE;
                }
                if (forceTime < 0) negativeVal = 1;
                break;
            case 'f':
                intData.appliedForceZ = parse_float(optarg, &error);
                if (error == 1) {
                    fprintf(stderr, "Error parsing force.\n");
                    return EXIT_FAILURE;
                }
                break;
            case 'k':
                intData.k_stretch = parse_float(optarg, &error);
                if (error == 1) {
                    fprintf(stderr, "Error parsing constant k for stretching.\n");
                    return EXIT_FAILURE;
                }
                if (intData.k_stretch < 0) negativeVal = 1;
                break;
            case 'K':
                intData.k_pend = parse_float(optarg, &error);
                if (error == 1) {
                    fprintf(stderr, "Error parsing constant k for pending.\n");
                    return EXIT_FAILURE;
                }
                if (intData.k_pend < 0) negativeVal = 1;
                break;
            case 'J':
                intData.J = parse_float(optarg, &error);
                if (error == 1) {
                    fprintf(stderr, "Error parsing moment of inertia J.\n");
                    return EXIT_FAILURE;
                }
                if (intData.J < 0) negativeVal = 1;
                break;
            case 'g':
                intData.gee = parse_float(optarg, &error);
                if (error == 1) {
                    fprintf(stderr, "Error parsing gravitational acceleration.\n");
                    return EXIT_FAILURE;
                }
                if (intData.gee < 0) negativeVal = 1;
                break;
            case 'r':
                runtime = parse_float(optarg, &error);
                if (error == 1) {
                    fprintf(stderr, "Error parsing runtime.\n");
                    return EXIT_FAILURE;
                }
                if (runtime < 0) negativeVal = 1;
                break;
            case 'm':
                massFlag = parse_float(optarg, &error);
                if (error == 1) {
                    fprintf(stderr, "Error parsing mass.\n");
                    return EXIT_FAILURE;
                }
                if (massFlag < 0) negativeVal = 1;
                break;
            case 'p':
                intData.Rot_fric = parse_float(optarg, &error);
                if (error == 1) {
                    fprintf(stderr, "Error parsing rotational friction.\n");
                    return EXIT_FAILURE;
                }
                if (intData.Rot_fric < 0) negativeVal = 1;
                break;
            case 'c':
                intData.c = parse_float(optarg, &error);
                if (error == 1) {
                    fprintf(stderr, "Error parsing spring damping.\n");
                    return EXIT_FAILURE;
                }
                if (intData.c < 0) negativeVal = 1;
                break;
            default:
                printf("\nSimple flag simulator\nFlag size: %.2lf m x %.2lf m\n\n"
                       , HNODES*SPRINGL, VNODES*SPRINGL);
                printf("Usage: Flagsim -t <> -T <> -f <> -k <> -K <> -J <> "
                       "-g <> -r <> -m <> -p <> -c <>\n\n");
                printf("-t:    timestep (0.0001 s)\n"
                       "-T:    For how long force acts (1 s)\n"
                       "-f:    acting force in z-direction (0.01 N)\n"
                       "-k:    spring constant for tension (400 N/m)\n"
                       "-K:    spring constant for pending (3 N/m)\n"
                       "-J:    Moment of inertia for a flag assembly (0.1001 kgm²)\n"
                       "-g:    gravitational acceleration (1.622 m/s²)\n"
                       "-r:    runtime for simulation (25 s)\n"
                       "-m:    Mass of the flag (0.116 kg)\n"
                       "-p:    Rotational friction (0.005 Nm)\n"
                       "-c:    Spring damping (0.0001 Ns/m)\n\n");
                printf("Interactive keys:\n\n"
                       "F1:    Select view from the side\n"
                       "F2:    Select view from overhead\n"
                       "t:     Print simulation time to terminal\n"
                       "ESC:   Quit simulation\n");

                    return EXIT_FAILURE;
        }
    }
    if (negativeVal == 1) {
        fprintf(stderr, "Some parameters don't allow negative values.\n");
        return EXIT_FAILURE;
    }

    retVal += createRectangle(&intData.flag, VNODES, HNODES, massFlag);
    retVal += createRectangle(&intData.flagTmp, VNODES, HNODES, massFlag);

    retVal += createAssembly(&intData.assembly, VNODES, HNODES);
    retVal += createAssembly(&intData.assemblyTmp, VNODES, HNODES);

    /* Does the right thing even if EXIT_SUCCESS is defined !0 */
    if (retVal != 4*EXIT_SUCCESS) {
        fprintf(stderr, "Allocation failed!\n");
        return EXIT_FAILURE;
    }

    /* long-winded way to say alfaPos = 0.0 */
    temp = (intData.assembly[HNODES-1].pos[2]*0.0 + intData.assembly[HNODES-1].pos[0]
            *1.2192)/(HNODES*SPRINGL*sqrt(intData.assembly[HNODES-1].pos[0]
            *intData.assembly[HNODES-1].pos[0] + intData.assembly[HNODES-1].pos[2]
            *intData.assembly[HNODES-1].pos[2]));
    intData.alfaPos = acos( temp );
    intData.alfaVel = 0.0;

    /* Set connections between flag and assembly */
    for (i=0; i < HNODES; i++) {
        /* Horizontal connections */
        intData.flag[i].s[0] = &intData.assembly[i];
        intData.assembly[i].s[2] = &intData.flag[i];
        intData.flagTmp[i].s[0] = &intData.assemblyTmp[i];
        intData.assemblyTmp[i].s[2] = &intData.flagTmp[i];
    }
    /* Vertical connection */
    intData.flag[HNODES*(VNODES-1)].s[3] = &intData.assembly[HNODES+VNODES-1];
    intData.flagTmp[HNODES*(VNODES-1)].s[3] = &intData.assemblyTmp[HNODES+VNODES-1];
    intData.assembly[HNODES+VNODES-1].s[1] = &intData.flag[HNODES*(VNODES-1)];
    intData.assemblyTmp[HNODES+VNODES-1].s[1] = &intData.flagTmp[HNODES*(VNODES-1)];

    /* Amount of nodes in array */
    intData.fNodes = HNODES*VNODES;
    intData.aNodes = HNODES+VNODES;


    struct timespec timeNow, sleep;

    double timeBetweenSleeps, t1, t2;

    /* Init mutex */
    if ( pthread_mutex_init(&intData.lockData, NULL) != 0 ) {
        fprintf(stderr, "Mutex initialization failed\n");
        return EXIT_FAILURE;
    }

    /* Create rendering thread */
    threadData.exitFlag = 0;
    threadData.viewFlag = 0;
    threadData.simulationTime = &timeElapsed;
    threadData.flag = intData.flag;
    threadData.assembly = intData.assembly;
    threadData.lockPtr = &intData.lockData;
    pthread_create( &renderThread, NULL
                    , (void*) &drawGraphics, (void*) &threadData);

    while ( (timeElapsed < runtime) && (threadData.exitFlag == 0) ) {
        /* Run 1/120 secs in simulation before sleeping
         * calling nanosleep and clock_gettime every iteration
         * would create unnecessary overhead */

        clock_gettime(CLOCK_MONOTONIC, &timeNow);
        t1 = (double)timeNow.tv_sec + (double)timeNow.tv_nsec*1e-9;

        timeBetweenSleeps = 0.0;
        while(timeBetweenSleeps < (1.0/120.0)) {
            if (threadData.exitFlag != 0)
                break;
            if (timeElapsed > forceTime) {
                intData.appliedForceZ = 0.0;
            }
            /* leapfrog optimization might have bugs,
             * doesn't have exact same results as without it.
             * In a second look issue seems to be in applying force,
             * so a non-issue */

            /* optimization is questionable with damping */

           // intData.leapFrogFirstIteration = 1; /* Disables optimization */
            intLeapfrog(&intData, dt);
            timeElapsed += dt;
            timeBetweenSleeps += dt;
        }
        clock_gettime(CLOCK_MONOTONIC, &timeNow);
        t2 = (double)timeNow.tv_sec + (double)timeNow.tv_nsec*1e-9;
        t1 = t2-t1;

        sleep.tv_sec = 0; /* is always zero, unless targeting over 1 s sleep */
        sleep.tv_nsec = (int)(((1.0/120.0) - t1)*1e9);

        nanosleep(&sleep, NULL);
    }
    /* Loop might exit with condition exitflag == 0.
     * Make sure graphics thread exits. */
    threadData.exitFlag = 1;

    /* Graphics thread joins */
    pthread_join(renderThread, NULL);

    /* Failure checking would be meaningless at this point */
    pthread_mutex_destroy(&intData.lockData);

    free(intData.flag);
    free(intData.flagTmp);
    free(intData.assembly);
    free(intData.assemblyTmp);

    return EXIT_SUCCESS;
}
