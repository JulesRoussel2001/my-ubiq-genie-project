import { NetworkId, NetworkScene } from 'ubiq';
import { ApplicationController } from '../../components/application';
import { TextToSpeechService } from '../../services/text_to_speech/service';
import { TextGenerationService } from '../../services/text_generation/service';
import { AnimationsService } from '../../services/animations/service';
import { MessageReader } from '../../components/message_reader';
import path from 'path';
import { fileURLToPath } from 'url';

export class ConversationalAgent extends ApplicationController {
  components: {
    chatReader?: MessageReader;
    textGenerationService?: TextGenerationService;
    textToSpeechService?: TextToSpeechService;
    animationsService?: AnimationsService;
  } = {};
  targetPeer: string = '';
  animation: string = '';

  constructor(configFile: string = 'config.json') {
    super(configFile);
  }

  async start(): Promise<void> {
    this.registerComponents();
    this.log(
      `Services registered: ${Object.keys(this.components).join(', ')}`,
      'info'
    );

    try {
      await this.joinRoom();
      this.log('Room joined – now defining pipeline', 'info');
      this.definePipeline();
      this.log('Pipeline defined', 'info');
    } catch (err: any) {
      this.log(`Error joining room: ${err}`, 'error');
    }
  }

  registerComponents() {
    this.components.textGenerationService = new TextGenerationService(this.scene);
    this.components.textToSpeechService   = new TextToSpeechService(this.scene);
    this.components.animationsService     = new AnimationsService(this.scene);

    this.components.chatReader = new MessageReader(this.scene, 97);
  }

  definePipeline() {
      // on: for callbacks, but definePipeline is called only once
      //special communication for 97
    // Handle incoming UI chat
    // this.components.chatReader?.on(
    //   'message',
    //   (fromPeer: string, payload: { type: string; message?: string }) => {
      let UserText = "";
      let LLMResponse = "";
      this.components.chatReader?.on('data', (data: any) => {
        const payload = JSON.parse(data.message.toString());
        const text = (payload.message as string).trim();
        UserText = text + '\n';
        this.log(`[Pipeline] got UI chat: "${text}"`, 'info');
      
        // send directly to your LLM
        this.components.textGenerationService!
          .sendToChildProcess('default', text + '\n');
      });
      
      
	// Step 3: When we receive a response from the text generation service, we send it to the text to speech service
  // Call the function oin "data " so when data is call
	this.components.textGenerationService?.on('data', (data: Buffer, identifier: string) => {
		const response = data.toString();
    LLMResponse = response;
		this.log('Received text generation response from child process ' + identifier + ': ' + response, 'info');
		this.components.animationsService?.sendToChildProcess('default', UserText + '\n');
	});


    // Step 4: Animation information coming from Animation LLM
	this.components.animationsService?.on('data', (data: Buffer, identifier: string) => {
		const response = data.toString();
		this.log('Received text generation response from child process ' + identifier + ': ' + response, 'info');

		this.animation = response.trim();
		this.components.textToSpeechService?.sendToChildProcess('default', LLMResponse + '\n');
	});
	  
	this.components.textToSpeechService?.on('data', (data: Buffer, identifier: string) => {
		let response = data;

		this.scene.send(new NetworkId(96), {
			type: 'AnimationAudio',
			targetPeer: this.targetPeer,
			audioLength: data.length,
			animationTitle: this.animation
		});

		while (response.length > 0) {
      this.log('Length: ' + response.length);
			this.scene.send(new NetworkId(95), response.slice(0, 16000));
			response = response.slice(16000);
		}
	});

    // LLM → TTS + Animation + ChatMessage back to Unity
    /*this.components.textGenerationService?.on(
      'data',
      async (buf) => {
        const response = buf.toString().trim();
        this.log(`[Pipeline] LLM→response: "${response}"`, 'info');
        const message = response.startsWith('>') ? response.slice(1).trim() : response.trim();
        this.log(`[Pipeline] Cleaned LLM message: "${message}"`, 'info');

        this.scene.send(new NetworkId(97), {
          type: 'ChatMessage',
          message
        });

        // fire off TTS and Animation services in parallel
        const payload    = message + '\n';
        
      
        const ttsPromise = this.components.textToSpeechService!
          .sendToChildProcessAsync('default', payload);
        const animPromise = this.components.animationsService!
          .sendToChildProcessAsync('default', lastUserText);
        
          animPromise.then((animationBuffer) => {
            const animation = animationBuffer.toString().trim();
            this.log(`Animation chosen by LLM→AnimationsService: ${animation}`, 'info');
    
            this.scene.send(new NetworkId(96), {
                type: 'AnimationTrigger',
                animation
            });
        }).catch((err) => {
            this.log(`Error in Animation service: ${err.message}`, 'error');
        });
        
    });*/
  }
}

if (fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const absConfig = path.resolve(__dirname, './config.json');
  const app       = new ConversationalAgent(absConfig);
  app.start();
}