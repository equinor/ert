pipeline {
    agent { label 'scout-ci7' }
    environment {
        WORKING_DIR = sh(script: 'mktemp -d', , returnStdout: true).trim()
    }
    stages {
        stage('setup') {
            steps {
                echo "Started with WORKING_DIR: ${WORKING_DIR}"
                sh 'sh testjenkins.sh setup'
            }
        }
	stage('build libres') {
            steps {
                sh 'sh testjenkins.sh build_libecl'
                sh 'sh testjenkins.sh build_libres'
            }
        }
	stage('build res') {
            steps {
                sh 'sh testjenkins.sh build_res'
            }
        }
	stage('run ctest') {
            steps {
                sh 'sh testjenkins.sh run_ctest'
            }
        }
	stage('run pytest_equinor') {
            steps {
                sh 'sh testjenkins.sh run_pytest_equinor'
            }
        }
	stage('run pytest_normal') {
            steps {
                sh 'sh testjenkins.sh run_pytest_normal'
            }
        }
    }
}


