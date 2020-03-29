import React from 'react';
import { API_ROOT } from '../../api-config';
import { UsageSection } from '../UsageSection';
import { UsageCode } from '../UsageCode';
import SyntaxHighlight from '../highlight/SyntaxHighlight';
import { withRouter } from 'react-router-dom';
import Model from '../Model'
import OutputField from '../OutputField'
import { Accordion } from 'react-accessible-accordion';
import SaliencyMaps from '../Saliency'
import InputReductionComponent from '../InputReduction'
import HotflipComponent from '../Hotflip'
import { FormLabel } from '../Form';
import {
  GRAD_INTERPRETER,
  IG_INTERPRETER,
  SG_INTERPRETER,
  INPUT_REDUCTION_ATTACKER,
  HOTFLIP_ATTACKER
} from '../InterpretConstants'

// title of the page
const title = "Psychotherapy MISC Analysis"

const NAME_OF_INPUT_TO_ATTACK = "tokens"
const NAME_OF_GRAD_INPUT = "grad_input_1"
const MISC11_P_labels = ["Positive","Negative","Neutral"]
const MISC11_T_labels = ["Facilitate","Simple Reflection","Complex Reflection","Giving Info","Closed Question","Open Question","MI Adherent","MI Non-adherent"]

// Text shown in the UI
const description = (
  <span>
    <span>
    MISC labels indicate the conversational behavior of ....
    </span>
    <p>
      <b>Contributed by:</b> Tarun Sunkaraneni
    </p>
  </span>
);
const descriptionEllipsed = (
  <span> MISC labels indicate the conversational behavior of… </span>
);

const tasks = [
  {
    name: "Forecast",
    desc: "Forecasting the MISC code of the Agent given conversation context"
  },
  {
    name: "Categorize",
    desc: "Categorize the MISC code of the Agent given their utterance"
  }
]

const agents = [
  {
    name: "Therapist",
    desc: "Analyze therapist MISC codes"
  },
  {
    name: "Patient",
    desc: "Analyze patient MISC codes"
  }
]

const taskEndpoints = {
  "Forecast,Therapist": "forecast-therapist-misc",
  "Categorize,Therapist":  "categorize-therapist-misc",
  "Forecast,Patient": "forecast-patient-misc",
  "Categorize,Patient":  "categorize-patient-misc",
};


// Input fields to the model.
const fields = [
  {name: "sentence", type: "PSYCH_TEXT_AREA", placeholder: '', optional: true},
  {name: "task", label: "Task", type: "RADIO", options: tasks, optional: true}, 
  {name: "agent", label: "Agent", type: "RADIO", options: agents, optional: true},
  {name: "label", label: "Label (for Online Learning)", type: "TEXT_INPUT", placeholder: 'E.g. "2"', optional: true},
  {name: "saliency_label", label: "Interpretation Label", type: "TEXT_INPUT", placeholder: 'E.g. "2"', optional: true}
]

const getUrl = (task, agent, apiCall) => {
  task = task || tasks[0].name;
  agent = agent || agents[0].name;
  const endpoint = taskEndpoints[[task,agent]];
  return `${API_ROOT}/${apiCall}/${endpoint}?cache=false`
}

const apiUrl = ({task, agent}) => {
  return getUrl(task, agent, "predict")
}

const apiUrlInterpret = ({task, agent}) => {
  return getUrl(task, agent, "interpret")
}

const apiUrlAttack = ({task, agent}) => {
  return getUrl(task, agent, "attack")
}

const getGradData = ({ grad_input_1: gradInput1 }) => {
  return [gradInput1];
}

const MySaliencyMaps = ({interpretData, tokens, interpretModel, requestData}) => {
  let simpleGradData = undefined;
  let integratedGradData = undefined;
  let smoothGradData = undefined;
  if (interpretData) {
    simpleGradData = GRAD_INTERPRETER in interpretData ? getGradData(interpretData[GRAD_INTERPRETER]['instance_1']) : undefined
    integratedGradData = IG_INTERPRETER in interpretData ? getGradData(interpretData[IG_INTERPRETER]['instance_1']) : undefined
    smoothGradData = SG_INTERPRETER in interpretData ? getGradData(interpretData[SG_INTERPRETER]['instance_1']) : undefined
  }
  const inputTokens = [tokens];
  const inputHeaders = [<p><strong>Utterance:</strong></p>];
  const allInterpretData = {simple: simpleGradData, ig: integratedGradData, sg: smoothGradData};
  return <SaliencyMaps interpretData={allInterpretData} inputTokens={inputTokens} inputHeaders={inputHeaders} interpretModel={interpretModel} requestData={requestData} />
}

// const Attacks = ({attackData, attackModel, requestData}) => {
//   let hotflipData = undefined;
//   if (attackData && "hotflip" in attackData) {
//     hotflipData = attackData["hotflip"];
//     const [pos, neg] = hotflipData["outputs"]["probs"]
//     hotflipData["new_prediction"] = pos > neg ? 'Positive' : 'Negative'
//   }
//   let reducedInput = undefined;
//   if (attackData && "input_reduction" in attackData) {
//     const reductionData = attackData["input_reduction"];
//     reducedInput = {original: reductionData["original"], reduced: [reductionData["final"][0]]};
//   }
//   return (
//     <OutputField label="Model Attacks">
//       <Accordion accordion={false}>
//         <InputReductionComponent reducedInput={reducedInput} reduceFunction={attackModel(requestData, INPUT_REDUCTION_ATTACKER, NAME_OF_INPUT_TO_ATTACK, NAME_OF_GRAD_INPUT)} />
//         <HotflipComponent hotflipData={hotflipData} hotflipFunction={attackModel(requestData, HOTFLIP_ATTACKER, NAME_OF_INPUT_TO_ATTACK, NAME_OF_GRAD_INPUT)} />
//       </Accordion>
//     </OutputField>
//   )
// }

// What is rendered as Output when the user hits buttons on the demo.
const Output = ({ responseData, requestData, interpretData, interpretModel, attackData, attackModel}) => {
  var labels = requestData.agent.toLowerCase() === 'therapist' ? MISC11_T_labels : MISC11_P_labels;
  const predictions = responseData.probabilities;
  let t = requestData;
  const tokens = responseData['model_input_tokens'];

  // The RoBERTa-large model is very slow to be attacked
  // const attacks = model && model.includes('RoBERTa') ?
  //   " "
  // :
  //   <Attacks attackData={attackData} attackModel={attackModel} requestData={requestData}/>

  // The "Answer" output field has the models predictions. The other output fields are the
  // reusable HTML/JavaScript for the interpretation methods.
  return (
    <div className="model__content answer">
      <FormLabel>Predictions</FormLabel>
      {labels.map((label, index) => {
        return (
        <OutputField key={index}>
          {label} : {predictions[index]}%
        </OutputField>
        );
      })}
    <OutputField>
      <Accordion accordion={false}>
          <MySaliencyMaps interpretData={interpretData} tokens={tokens} interpretModel={interpretModel} requestData={requestData}/>
          {/* {attacks} */}
      </Accordion>
    </OutputField>
  </div>
  );
}

// Examples the user can choose from in the demo
const examples = [
  { sentence: "Therapist: Why do you think you worry about Bryan getting married\nPatient: I don't know, I guess i'm just scared of losing my best friend\nTherapist: But as a best friend wouldn\'t you want what\'s best for Bryan?\nPatient: I know it is what\s best for him but i can\t help but feel foolish at times\nTherapist: I encourage you to look beyond yourself as being happy for loved ones can make you happy." }]

const usage = (
  <React.Fragment>
    <UsageSection>
      <h3>Prediction</h3>
      <h5>On the command line (bash):</h5>
      <UsageCode>
        <SyntaxHighlight language="bash">
          {`echo '{"sentence": "a very well-made, funny and entertaining picture."}' | \\
allennlp predict https://storage.googleapis.com/allennlp-public-models/sst-roberta-large-2020.02.17.tar.gz -`}
        </SyntaxHighlight>
      </UsageCode>
      <h5>As a library (Python):</h5>
      <UsageCode>
        <SyntaxHighlight language="python">
          {`from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/sst-roberta-large-2020.02.17.tar.gz")
predictor.predict(
  sentence="a very well-made, funny and entertaining picture."
)`}
        </SyntaxHighlight>
      </UsageCode>
    </UsageSection>
    <UsageSection>
      <h3>Evaluation</h3>
      <UsageCode>
        <SyntaxHighlight language="python">
          {`allennlp evaluate \\
  https://storage.googleapis.com/allennlp-public-models/sst-roberta-large-2020.02.17.tar.gz \\
  https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt`}
        </SyntaxHighlight>
      </UsageCode>
    </UsageSection>
    <UsageSection>
      <h3>Training</h3>
      <UsageCode>
        <SyntaxHighlight language="python">
          allennlp train training_config/basic_stanford_sentiment_treebank.jsonnet -s output_path
        </SyntaxHighlight>
      </UsageCode>
    </UsageSection>
  </React.Fragment>
)

// A call to a pre-existing model component that handles all of the inputs and outputs. We just need to pass it the things we've already defined as props:
const modelProps = {apiUrl, apiUrlInterpret, apiUrlAttack, title, description, descriptionEllipsed, fields, examples, Output, usage}
export default withRouter(props => <Model {...props} {...modelProps}/>)
