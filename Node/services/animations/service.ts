import { ServiceController } from '../../components/service';
import { NetworkScene } from 'ubiq';
import nconf from 'nconf';

export class AnimationsService extends ServiceController {
    constructor(scene: NetworkScene) {
        super(scene, 'AnimationsService');

        console.log(
            'animations_preprompt:',
            nconf.get('animations_preprompt'),
            'animations_prompt_suffix:',
            nconf.get('animations_prompt_suffix')
        );

        this.registerChildProcess('default', 'python', [
            '-u',
            '../../services/animations/animations_openai.py',
            '--preprompt',
            nconf.get('animations_preprompt') || '',
            '--prompt_suffix',
            nconf.get('animations_prompt_suffix') || '',
        ]);
    }
}